import os, csv, random, time
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from rwm.config.config import MEMORY_BATCH
from rwm.models.rwm_deterministic.model import ReducedWorldModel
from rwm.models.controller.model import Controller
from rwm.policies.controller_policy import ControllerPolicy
from rwm.types import Rollout
from rwm.utils.preprocess_observation import preprocess_obs
from rwm.utils.rollout_simulator import RolloutSimulator
from rwm.utils.behavior_memory import BehaviorMemory


class ControllerTrainer:
	def __init__(
		self,
		model: ReducedWorldModel,
        model_ckpt: str,
		controller: Controller,
		scenarios_dirs: List[str],
		out_dir: str = "runs/controller",
		memory_dir: str = "runs/controller/memory",
		memory_size: int = 1000,
		warmup_steps: int = 5,
		rollout_len: int = 20,
		noise_std: float = 0.3,
		lr: float = 1e-3,
		device: str = "cuda"
	) -> None:
		self.out_dir = out_dir
		self.device = device
		self.model = model.to(device).eval()
		self.model.load_state_dict(torch.load(model_ckpt))

		self.controller = controller.to(device)
		self.policy = ControllerPolicy(
			controller=self.controller,
			noise_std=noise_std
		)

		self.scenarios_dirs = scenarios_dirs
		self.simulator = RolloutSimulator(
			model=self.model,
			policy=self.policy,
			warmup_steps=warmup_steps,
			rollout_len=rollout_len,
			device=device
		)

		self.optimizer = optim.Adam(self.controller.parameters(), lr=lr)

		self.memory = BehaviorMemory(situation_dir=memory_dir, max_size=memory_size)
		self.memory_path = os.path.join(out_dir, "behavior_memory.pkl")
		if os.path.exists(self.memory_path):
			self.memory.load(self.memory_path)



	def _select_topk(self, rollouts: List[Rollout], k: int) -> List[Rollout]:
		""" Sort by cumulative reward and return top k. """
		positive_rollouts = [r for r in rollouts if r.cum_reward > 0]
		if not positive_rollouts:
			return []
		return sorted(positive_rollouts, key=lambda x: x.cum_reward, reverse=True)[:k]


	def _optimize_topk(self, top_rollouts: List[Rollout]) -> float:
		""" Backprop on the top trajectories; returns accumulated loss. """
		self.controller.train()
		self.optimizer.zero_grad()

		losses: List[Tensor] = []

		for r in top_rollouts:
			for h_t, a_t in zip(r.latents, r.real_acts):
				pred = self.controller(h_t)
				loss = nn.functional.mse_loss(pred, a_t.detach())
				losses.append(loss)

		total_loss_tensor = torch.stack(losses).sum()
		total_loss_tensor.backward()						# type: ignore
		self.optimizer.step()								# type: ignore

		return float(total_loss_tensor.item())
	

	def train(
		self,
		n_rollouts: int = 100,
		top_k: int = 10,
		epochs: int = 20
	) -> None:
		os.makedirs(self.out_dir, exist_ok=True)
		metrics_path = os.path.join(self.out_dir, "metrics.csv")

		# write header
		with open(metrics_path, "w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow([
                "epoch", "avg_reward", "loss_rollouts", "loss_memory",
                "positives", "memory_size", "duration_s",
            ])
			
		pbar: tqdm[int] = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
		for epoch in pbar:
			t0 = time.time()

			# 1) generate imagined rollouts
			scenario = random.choice(self.scenarios_dirs)
			rollouts = self.simulator.generate_rollouts(scenario, n_rollouts)			

			# 2) Add positive situations to memory
			pos_count: int = 0
			for r in rollouts:
				if r.cum_reward > 0:
					self.memory.add(r.latents[0], r.cum_reward, r.real_obs, r.real_acts)
					pos_count += 1
			self.memory.save(self.memory_path)

			# 3) Update controller on rollouts
			top_rollouts  = self._select_topk(rollouts, top_k)
			loss_rollouts = self._optimize_topk(top_rollouts)

			# 4) Replay train on memory
			loss_memory = self.train_on_memory(MEMORY_BATCH)
			mem_size = len(self.memory)

			avg_reward = float(np.mean([r.cum_reward for r in top_rollouts])) if top_rollouts else 0.0
			duration = time.time() - t0

			pbar.set_postfix({							# type: ignore
                "avg_r":    f"{avg_reward:.2f}",
                "l_roll":   f"{loss_rollouts:.4f}",
                "l_mem":    f"{loss_memory:.4f}",
                "pos":      f"{pos_count}/{n_rollouts}",
                "mem":      f"{mem_size}",
                "time(s)":  f"{duration:.1f}"
            })

			with open(metrics_path, "a", newline="") as f:
				writer = csv.writer(f)
				writer.writerow([
					epoch,
					f"{avg_reward:.2f}",
					f"{loss_rollouts:.4f}",
					f"{loss_memory:.4f}",
					pos_count,
					mem_size,
					f"{duration:.2f}",
				])

		# 5) final save of controller weights
		controller_path = os.path.join(self.out_dir, "controller.pt")
		torch.save(self.controller.state_dict(), controller_path)
		pbar.close()
		print("Controller saved to controller.pt")


	def train_on_memory(self, batch_size: int) -> float:
		"""
		Sample up to batch_size behaviors from memory and do one gradient step.
		Returns the total training loss on that memory batch.
		"""
		keys = self.memory.sample(batch_size)
		if not keys:
			return 0.0

		losses: List[Tensor] = []
		for key in keys:
			path = self.memory.get_situation_path(key)
			obs_seq, act_seq = self.memory.load_obs_and_act(path)

			# warmup to get latent states
			h = torch.zeros(1, self.model.world_rnn.rnn_cell.hidden_size, device=self.device)
			c = torch.zeros_like(h)
			latents: List[Tuple[Tensor, Tensor]] = []
			for i in range(len(obs_seq)):
				img_t = preprocess_obs(obs_seq[i]).unsqueeze(0).to(self.device)
				a_prev = torch.from_numpy(act_seq[i:i+1]).to(self.device).float()			# type: ignore
				h, c, *_ = self.model.forward(img=img_t,
												a_prev=a_prev,
												h_prev=h, c_prev=c,
												force_keep_input=True)
				latents.append((h, a_prev))

			# compute MSE loss on the rollout’s actions
			for h_t, a_t in latents:
				pred = self.controller(h_t)
				losses.append(nn.functional.mse_loss(pred, a_t.detach()))

		# single backward pass
		total_loss = torch.stack(losses).sum()
		self.controller.train()
		self.optimizer.zero_grad()
		total_loss.backward()					# type: ignore
		self.optimizer.step()					# type: ignore
		return float(total_loss.item())