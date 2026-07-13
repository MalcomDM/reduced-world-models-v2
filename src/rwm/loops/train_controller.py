import os, csv, random, time
import numpy as np
from tqdm import tqdm
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from rwm.config.config import MEMORY_BATCH, POSITIVE_THRESHOLD
from rwm.models.rwm.model import ReducedWorldModel
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
		memory_size: int = 1000,
		warmup_steps: int = 5,
		rollout_len: int = 20,
		noise_std: float = 0.3,
		lr: float = 1e-3,
		device: str = "cuda",
		simulator: Optional[RolloutSimulator] = None,
	) -> None:
		self.out_dir = out_dir
		self.device = device
		self.model = model.to(device).eval()
		if model_ckpt:
			self.model.load_state_dict(torch.load(model_ckpt))

		self.controller = controller.to(device)
		self.policy = ControllerPolicy(
			controller=self.controller,
			noise_std=noise_std
		)

		self.scenarios_dirs = scenarios_dirs
		self.simulator = simulator or RolloutSimulator(
			model=self.model,
			policy=self.policy,
			warmup_steps=warmup_steps,
			rollout_len=rollout_len,
			device=device
		)

		self.optimizer = optim.Adam(self.controller.parameters(), lr=lr)

		mem_dir = os.path.join(self.out_dir, "memory")
		self.memory = BehaviorMemory(situation_dir=mem_dir, max_size=memory_size)
		self.memory_path = os.path.join(out_dir, "behavior_memory.pkl")
		if os.path.exists(self.memory_path):
			self.memory.load(self.memory_path)


	def _filter_positive(self, rollouts: List[Rollout]) -> List[Rollout]:
		"""Return only those rollouts whose cumulative reward exceeds the threshold."""
		return [r for r in rollouts if r.cum_reward > POSITIVE_THRESHOLD]


	def _select_topk(self, rollouts: List[Rollout], k: int) -> List[Rollout]:
		""" Sort by cumulative reward and return top k. """
		positives = self._filter_positive(rollouts)
		if not positives:
			return []
		return sorted(positives, key=lambda x: x.cum_reward, reverse=True)[:k]


	def _optimize_topk(self, top_rollouts: List[Rollout]) -> float:
		""" Backprop on the top trajectories; returns accumulated loss. """
		self.controller.train()
		self.optimizer.zero_grad()

		losses: List[Tensor] = []

		if not top_rollouts: return 0.0

		for r in top_rollouts:
			for h_t, a_t in zip(r.latents, r.sim_acts):
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
			rollouts: List[Rollout] = []
			scenario = random.choice(self.scenarios_dirs)
			rollouts.extend( self.simulator.generate_rollouts(scenario, n_rollouts) )

			# 2) Add positive situations to memory
			pos_rollouts = self._filter_positive(rollouts)
			pos_count = len(pos_rollouts)
			for r in pos_rollouts:
				self.memory.add(r.latents[0], r.cum_reward, r.real_obs, r.real_acts)
			self.memory.save(self.memory_path)

			# 3) Update controller on rollouts
			top_rollouts  = self._select_topk(pos_rollouts, top_k)
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
			
			del rollouts, top_rollouts, loss_memory, loss_rollouts
			torch.cuda.empty_cache()

		# 5) final save of controller weights
		controller_path = os.path.join(self.out_dir, "controller.pt")
		torch.save(self.controller.state_dict(), controller_path)
		pbar.close()
		print("Controller saved to controller.pt")


	def train_on_memory(self, batch_size: int) -> float:
		"""Train controller from behavior-memory replay.

		.. deprecated::
		   This method calls the legacy LSTM interface (``h_prev``,
		   ``c_prev``, ``world_rnn.rnn_cell``) which is incompatible
		   with the current ``CausalTransformer``-based world model.

		   The memory-based controller training path will be replaced
		   by Actor-Critic training in Stage 4 and is not maintained
		   during Stages 1-3.
		"""
		raise NotImplementedError(
			"train_on_memory uses the legacy LSTM world_rnn interface "
			"which is not available in the CausalTransformer-based "
			"ReducedWorldModel.\n\n"
			"This path will be replaced by Actor-Critic training "
			"(Stage 4) and is intentionally disabled."
		)