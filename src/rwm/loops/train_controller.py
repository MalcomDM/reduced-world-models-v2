import os, csv, random
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from rwm.models.rwm_deterministic.model import ReducedWorldModel
from rwm.models.controller.model import Controller
from rwm.policies.controller_policy import ControllerPolicy
from rwm.utils.rollout_simulator import RolloutSimulator


Rollout = Tuple[List[Tensor], List[Tensor], List[float], float]


class ControllerTrainer:
	def __init__(
		self,
		model: ReducedWorldModel,
        model_ckpt: str,
		controller: Controller,
		scenarios_dirs: List[str],
		warmup_steps: int = 5,
		rollout_len: int = 20,
		noise_std: float = 0.1,
		lr: float = 1e-3,
		device: str = "cuda"
	) -> None:
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



	def _select_topk(self, rollouts: List[Rollout], k: int) -> List[Rollout]:
		""" Sort by cumulative reward and return top k. """
		return sorted(rollouts, key=lambda x: x[3], reverse=True)[:k]


	def _optimize_topk(self, top_rollouts: List[Rollout]) -> float:
		""" Backprop on the top trajectories; returns accumulated loss. """
		total_loss = 0.0
		self.controller.train()
		self.optimizer.zero_grad()

		for latents, actions, _, _ in top_rollouts:
			for h_t, a_t in zip(latents, actions):
				pred = self.controller(h_t)
				loss = nn.functional.mse_loss(pred, a_t.detach())
				loss.backward()										# type: ignore
				total_loss += float(loss.item())

		self.optimizer.step()										# type: ignore
		return total_loss
	

	def train(
		self,
		n_rollouts: int = 100,
		top_k: int = 10,
		epochs: int = 20,
		out_dir: str = "runs/controller"
	) -> None:
		os.makedirs(out_dir, exist_ok=True)
		metrics_path = os.path.join(out_dir, "metrics.csv")

		# write header
		with open(metrics_path, "w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(["epoch", "avg_reward", "loss"])
			
		pbar: tqdm[int] = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
		for epoch in pbar:
			rollouts = []
			for _ in range(n_rollouts):
				scenario = random.choice(self.scenarios_dirs)
				rollouts = self.simulator.generate_rollouts(scenario, n_rollouts)			# 1) generate imagined rollouts
				top_rollouts = self._select_topk(rollouts, top_k)									# 2) pick the best k
				total_loss = self._optimize_topk(top_rollouts)										# 3) update controller on those

				avg_reward = float(np.mean([r[3] for r in top_rollouts]))
				pbar.set_postfix({							# type: ignore
					"avg_reward": f"{avg_reward:.2f}",
					"loss":       f"{total_loss:.4f}"
				})

				with open(metrics_path, "a", newline="") as f:
					writer = csv.writer(f)
					writer.writerow([epoch, f"{avg_reward:.2f}", f"{total_loss:.4f}"])

		torch.save(self.controller.state_dict(), "controller.pt")	# 4) save final weights
		pbar.close()
		print("Controller saved to controller.pt")