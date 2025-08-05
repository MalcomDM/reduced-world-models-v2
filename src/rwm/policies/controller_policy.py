import numpy as np
from numpy.typing import NDArray
from typing import Optional

import torch
from torch import Tensor

from rwm.policies.base_policy import BasePolicy
from rwm.models.controller.model import Controller


class ControllerPolicy(BasePolicy):
	def __init__(
		self,
		controller: Controller,
		noise_std: float = 0.0
	):
		self.controller = controller.eval()
		self.noise_std  = noise_std
		self.reset()


	def act(
		self,
		obs: Optional[NDArray[np.float32]],
		prev_action: Optional[NDArray[np.float32]] = None
	) -> NDArray[np.float32]:
		raise RuntimeError(
			"ControllerPolicy.act() is not supported. "
			"Use act_from_rwm_state(h: torch.Tensor) instead."
		)


	def act_from_rwm_state(self, h: Tensor) -> Tensor:
		"""
		Given a world-model hidden state `h` (1×WRNN_HIDDEN_DIM),
		produce a (1×3) action tensor, with optional noise.
		"""
		with torch.no_grad():
			a = self.controller(h)               # raw output
			if self.noise_std > 0:
				a = a + torch.randn_like(a) * self.noise_std
			# clamp each channel
			steer = a[:, 0].clamp(-1.0, 1.0).unsqueeze(1)
			gas   = a[:, 1].clamp( 0.0, 1.0).unsqueeze(1)
			brk   = a[:, 2].clamp( 0.0, 1.0).unsqueeze(1)
			return torch.cat([steer, gas, brk], dim=1)