import random
import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch import Tensor

from rwm.config.config import ACTION_DIM, VALUES_DIM, SEQ_LEN
from rwm.models.rwm.model import ReducedWorldModel
from rwm.policies.controller_policy import ControllerPolicy
from rwm.types import Rollout, WorldModelOutput
from rwm.utils.load_rollouts_from_scenario import load_rollouts_from_scenario
from rwm.utils.preprocess_observation import preprocess_obs
from rwm.utils.history_buffer import HistoryBuffer



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
		self.input_dim    = VALUES_DIM + ACTION_DIM


	def warmup_state(
		self,
		obs_seq: NDArray[np.uint8],
		act_seq: NDArray[np.float32]
	) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
		B = 1
		device = self.device
		assert self.warmup_steps <= len(obs_seq), "warmup_steps > len(obs_seq)"

		buf = HistoryBuffer(
			max_seq_len=SEQ_LEN,
			input_dim=self.input_dim,
			device=device,
		)
		prev_action = torch.zeros(1, ACTION_DIM, device=self.device)

		world_state: Tensor = torch.zeros(
			1, getattr(self.model.world_hd, "d_model", 64), device=device,
		)

		with torch.no_grad():
			for i in range(self.warmup_steps):
				obs = obs_seq[i]
				# prev_action is action[i-1] (zeros at i=0).
				# current_action is action[i] (used for reward head).
				current_action_np = act_seq[i]
				current_action = (
					torch.from_numpy(current_action_np[None]).float().to(device)
				)

				img_t = preprocess_obs(obs).to(self.device)

				out: WorldModelOutput = self.model(
					img=img_t,
					prev_action=prev_action,
					current_action=current_action,
					history=buf.history,
					lengths=buf.valid_lengths,
					force_keep_input=True,
				)
				world_state = out.world_state
				buf = HistoryBuffer.from_history(
					SEQ_LEN, self.input_dim, out.history, out.lengths,
				)
				# Next step's prev_action is this step's current_action.
				prev_action = current_action

		h_spatial_last = self.model.generate_spatial_rep(
			preprocess_obs(obs_seq[self.warmup_steps - 1]).to(self.device),
		)
		return buf.history, buf.valid_lengths, world_state, h_spatial_last


	def imagine_rollout(
		self,
		history: Tensor,
		lengths: Tensor,
		world_state: Tensor,
		h_spatial_last: Tensor
	) -> Tuple[List[Tensor], List[Tensor], List[float]]:
		"""
		Perform imagined rollout_len steps using the causal Transformer.
		First step sees the real h_spatial; all subsequent steps see zero_spatial.
		"""
		latent_hist: List[Tensor] = []
		action_hist: List[Tensor] = []
		reward_hist: List[float] = []

		zero_spatial = torch.zeros_like(h_spatial_last, device=self.device)

		buf = HistoryBuffer.from_history(
			SEQ_LEN, self.input_dim, history, lengths,
		)
		ws = world_state

		with torch.no_grad():
			for i in range(self.rollout_len):
				latent_hist.append(ws)
				a_t = self.policy.act_from_rwm_state(ws).to(self.device)
				action_hist.append(a_t)

				x_in = h_spatial_last if i == 0 else zero_spatial
				token_t = torch.cat([x_in, a_t], dim=-1).unsqueeze(1)

				buf.append(token_t)
				ws = self.model.world_hd(
					buf.history, lengths=buf.valid_lengths,
				)
				_shared, r_pred = self.model.controller(
					ws, a_t,
				)
				reward_hist.append(float(r_pred.squeeze(-1).item()))
				ws = ws.detach()

		return latent_hist, action_hist, reward_hist


	def _generate_one(self, scenarios_dir: str, obs_seq: NDArray[np.uint8], act_seq: NDArray[np.float32]) -> Rollout:
		# exactly the body of your old loop, but for one rollout
		max_start = len(obs_seq) - (self.warmup_steps + self.rollout_len)
		start = random.randint(0, max_start)

		seg_obs = obs_seq[start : start + self.warmup_steps + self.rollout_len]
		seg_acts = act_seq[start : start + self.warmup_steps + self.rollout_len]

		hist, lens, ws, h_sp_last = self.warmup_state(seg_obs, seg_acts)
		latents, sim_acts, rewards = self.imagine_rollout(hist, lens, ws, h_sp_last)
		cum = float(sum(rewards))

		return Rollout(
			real_obs=seg_obs,
			real_acts=seg_acts,
			latents=latents,
			sim_acts=sim_acts,
			rewards=rewards,
			cum_reward=cum
		)


	def generate_rollouts(self, scenarios_dir: str, n: int) -> List[Rollout]:
		"""Sample `n` segments from real `.npz` files, warm up, then imagine."""

		results: List[Rollout] = []
		obs_seq, act_seq = load_rollouts_from_scenario(scenarios_dir)
		with ThreadPoolExecutor() as exe:
			futures = [exe.submit(self._generate_one, scenarios_dir, obs_seq, act_seq) for _ in range(n)]
			for fut in tqdm(as_completed(futures), total=n, desc="Generating rollouts"):
				results.append(fut.result())
		return results
