import os, pickle
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from numpy.typing import NDArray

import torch
from torch import Tensor

from rwm.config.config import WRNN_HIDDEN_DIM
from rwm.utils.preprocess_observation import preprocess_obs



class BehaviorMemory:
	def __init__(self, situation_dir: str, max_size: int = 1000):
		self.max_size = max_size
		self.situation_dir = Path(situation_dir)
		self.situation_dir.mkdir(parents=True, exist_ok=True)
		self.states: Dict[str, Dict[str, Any]] = {}


	def __len__(self) -> int:
		return len(self.states)


	def hash_state(self, state: Tensor) -> str:
		"""Hash the initial latent state to group similar situations."""
		return torch.round(state * 100).to(torch.int).cpu().numpy().tobytes().hex()		# type: ignore
	

	def _save_situation(self, filename: str, obs: NDArray[np.uint8], actions: NDArray[np.float32] ) -> None:
		path = self.situation_dir / filename
		np.savez(path, obs=obs, actions=actions)


	def add(
			self,
			initial_state: Tensor,
			cum_reward: float,
			real_obs: NDArray[np.uint8],
			real_acts: NDArray[np.float32]
	) -> None:
		"""
		Store or replace the best behavior for the given initial latent state.
		real_obs/real_acts define the warmup segment to re-simulate later.
		"""
		key = self.hash_state(initial_state)
		fname = key[:16]
		npz_name = f"{fname}.npz"
		should_replace = (key not in self.states) or (cum_reward > self.states[key]["cum_reward"])

		if should_replace:
			self._save_situation(npz_name, real_obs, real_acts)
			prev_count = self.states.get(key, {}).get("reencode_count", 0)
			self.states[key] = {
				"file": npz_name,
				"cum_reward": cum_reward,
				"reencode_count": prev_count
			}

			if len(self.states) > self.max_size:
				worst_key = min(self.states, key=lambda k: self.states[k]["cum_reward"])
				try: (self.situation_dir / self.states[worst_key]["file"]).unlink()
				except FileNotFoundError: pass
				del self.states[worst_key]



	def sample(self, k: int) -> List[str]:
		keys = list(self.states.keys())
		if not keys: return []
		idxs = torch.randperm(len(keys))[:min(k, len(keys))]
		return [keys[i] for i in idxs]
	

	def get_rollout_info(self, key: str) -> Optional[Dict[str, Any]]:
		return self.states.get(key)
	

	def get_situation_path(self, key: str) -> Path:
		return self.situation_dir / self.states[key]["file"]


	def save(self, path: str) -> None:
		os.makedirs(os.path.dirname(path), exist_ok=True)
		with open(path, "wb") as f:
			pickle.dump(self.states, f)


	def load(self, path: str) -> None:
		with open(path, "rb") as f:
			self.states = pickle.load(f)

	def load_obs_and_act(self, path:Path) -> Tuple[NDArray[np.uint8], NDArray[np.float32]]:
		with np.load(path) as data:
			obs_seq = data["obs"]
			act_seq = data["actions"]
		return obs_seq, act_seq



	def recompute_keys(self, model: torch.nn.Module) -> None:
		new_states: Dict[str, Dict[str, Any]] = {}
		device = next(model.parameters()).device

		for _old_key, info in self.states.items():
			path = self.situation_dir / info["file"]
			obs_seq, act_seq = self.load_obs_and_act(path)

			h = torch.zeros(1, WRNN_HIDDEN_DIM, device=device)
			c = torch.zeros_like(h)

			for i in range(len(obs_seq)):
				img_t = preprocess_obs(obs_seq[i]).unsqueeze(0).to(device)
				a_t = torch.from_numpy(act_seq[i:i+1]).to(device)										# type: ignore

				h, c, *_ = model.forward(img_t, a_t, h, c, force_keep_input=True)
			
			new_key = self.hash_state(h)
			prev_reencode = info.get("reencode_count", 0)

			if new_key in new_states:
				if info["cum_reward"] > new_states[new_key]["cum_reward"]:
					new_states[new_key] = {
						**info,
						"reencode_count": new_states[new_key].get("reencode_count", prev_reencode) + 1
					}
				else:
					new_states[new_key]["reencode_count"] += 1
			else:
				new_states[new_key] = {
					**info,
					"reencode_count": prev_reencode + 1
				}

		self.states = new_states

