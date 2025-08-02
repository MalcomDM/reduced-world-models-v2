import numpy as np
from numpy.typing import NDArray
from typing import Optional


class BasePolicy:
	def reset(self) -> None:
		"""Optional reset before a new rollout."""
		pass

	def act(
		self,
		obs: NDArray[np.float32],
		prev_action: Optional[NDArray[np.float32]] = None
	) -> NDArray[np.float32]:
		raise NotImplementedError