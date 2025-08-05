import random
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

import torch
from torch import Tensor

from rwm.models.rwm_deterministic.model import ReducedWorldModel
from rwm.policies.controller_policy import ControllerPolicy
from rwm.utils.load_rollouts_from_scenario import load_rollouts_from_scenario
from rwm.utils.preprocess_observation import preprocess_obs  # your full model



Rollout = Tuple[List[Tensor], List[Tensor], List[float], float]


class RolloutSimulator:
	def __init__(
		self,
		model: ReducedWorldModel,
		policy: ControllerPolicy,
		warmup_steps: int = 5,
		rollout_len: int = 20,
		device: str = "cuda"
	) -> None:
		self.device       = device
		self.model        = model.to(device).eval()
		self.policy       = policy
		self.warmup_steps = warmup_steps
		self.rollout_len  = rollout_len


	def warmup_state(
		self,
		obs_seq: NDArray[np.uint8],
		act_seq: NDArray[np.float32]
	) -> Tuple[Tensor, Tensor, Tensor]:
		"""
		Run `warmup_steps` through full model pipeline to get initial h, c, and h_spatial.
		Returns last h, last c, last spatial token, and last action.
		"""
		h = torch.zeros(1, self.model.world_rnn.rnn_cell.hidden_size, device=self.device)
		c = torch.zeros_like(h)

		first_img = preprocess_obs(obs_seq[0]).to(self.device)
		h_spatial: Tensor = self.model.generate_spatial_rep(first_img)

		for i in range(self.warmup_steps):
			obs = obs_seq[i]
			a_prev = act_seq[i]

			img_t = preprocess_obs(obs).to(self.device)
			a_t   = torch.from_numpy(a_prev[None]).to(self.device)	# type: ignore
			
			h, c, *_ = self.model.forward(
				img=img_t, a_prev=a_t, h_prev=h, c_prev=c,
				force_keep_input=True
			)
			h_spatial = self.model.generate_spatial_rep(img_t)

		return h, c, h_spatial


	def imagine_rollout(
		self,
		h: Tensor,
		c: Tensor,
		h_spatial: Tensor
	) -> Tuple[List[Tensor], List[Tensor], List[float]]:
		"""
		Perform imagined rollout_len steps using only WorldRNN.
		First step sees the real h_spatial; all subsequent steps see zero_spatial.
		"""
		latent_hist: List[Tensor] = []
		action_hist: List[Tensor] = []
		reward_hist: List[float] = []

		zero_spatial = torch.zeros_like(h_spatial, device=self.device)

		for i in range(self.rollout_len):
			latent_hist.append(h)
			a_t = self.policy.act_from_rwm_state(h).to(self.device)
			action_hist.append(a_t)

			x_spatial = h_spatial if i == 0 else zero_spatial

			h, c, r = self.model.world_rnn(
				h_prev=h, c_prev=c,
				x_spatial=x_spatial,
				a_prev=a_t,
				force_keep_input=True
			)
			reward_hist.append(r.item())

		return latent_hist, action_hist, reward_hist


	def generate_rollouts(self, scenarios_dir: str, n: int) -> List[Rollout]:
		"""Sample `n` segments from real `.npz` files, warm up, then imagine."""

		results: List[Rollout] = []
		for _ in range(n):
			obs_seq, act_seq = load_rollouts_from_scenario(scenarios_dir)
			max_start = len(obs_seq) - (self.warmup_steps + self.rollout_len)
			start = random.randint(0, max_start)

			seg_obs = obs_seq[start : start + self.warmup_steps + self.rollout_len]
			seg_act = act_seq[start : start + self.warmup_steps + self.rollout_len]

			h, c, h_spatial = self.warmup_state(seg_obs, seg_act)
			latents, actions, rewards = self.imagine_rollout(h, c, h_spatial)
			results.append((latents, actions, rewards, sum(rewards)))
		return results
