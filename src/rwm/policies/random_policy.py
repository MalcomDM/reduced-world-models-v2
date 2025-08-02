import numpy as np
from numpy.typing import NDArray
from typing import Optional

from .base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    
	def __init__(self, smooth: bool = False, noise_scale: float = 0.2):
		self.smooth = smooth
		self.noise_scale = noise_scale
		self.prev_action = np.zeros(3, dtype=np.float32)


	def reset(self) -> None:
		self.prev_action = np.zeros(3, dtype=np.float32)


	def act(
		self, obs: NDArray[np.float32],
		prev_action: Optional[NDArray[np.float32]] = None
	) -> NDArray[np.float32]:
		if self.smooth:
			return self._smooth_action()
		return self._random_action()


	def _smooth_action(self) -> NDArray[np.float32]:
		noise = np.random.normal(0, self.noise_scale, size=3).astype(np.float32)
		self.prev_action = np.clip(self.prev_action + noise, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
		return self.prev_action


	def _random_action(self) -> NDArray[np.float32]:
		steer = np.clip(np.random.normal(0, 0.5), -1.0, 1.0)
		gas = np.random.beta(2, 2)
		brake = np.random.binomial(1, 0.05) * np.random.rand()
		return np.array([steer, gas, brake], dtype=np.float32)