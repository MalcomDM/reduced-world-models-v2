import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple

from rwm.config.config import K




class TopKGumbelSelector(nn.Module):
	"""
	Differentiable Top-K via Gumbel-Softmax + STE.
	forward(logits) -> mask_soft, indices
		- mask_soft: (B,N) values in [0,1], sums≈1
		- indices:  (B,K) hard indices of selected tokens
	"""
	def __init__(self, k: int = K, temp: float = 1.0) -> None:
		super().__init__()											# type: ignore[reportUnknownMemberType]
		self.k = k
		self.temp = temp


	@staticmethod
	def sample_gumbel(shape: Tuple[int, ...], device: torch.device, eps: float=1e-20) -> Tensor:
		"""Sample Gumbel(0,1) noise with the same dtype/device as inputs."""
		U = torch.rand(shape, device=device)
		return -torch.log(-torch.log(U + eps) + eps)


	def forward(self, logits:Tensor) -> Tuple[Tensor, Tensor]:
		"""
		logits: (B, N)
		returns:
			mask_soft: (B, N) -- during training use this to multiply tokens
			indices:   (B, K) -- the hard top-K indices (for inference or gather)
		"""
		B, N = logits.shape

		# 1) Decide which logits to top-k on
		if self.training:
			# only inject noise during training
			gumbel_noise = self.sample_gumbel(shape=(B,N), device=logits.device)
			topk_logits  = (logits + gumbel_noise) / self.temp
		else:
			# deterministic top-K in eval
			topk_logits = logits

		topk_indices = topk_logits.topk(self.k, dim=1).indices # (B, K)				# 2) Get hard top-k indices
		mask_hard = torch.zeros_like(logits).scatter_(1, topk_indices, 1.0)			# 3) Build hard mask
		mask_soft = F.softmax(logits / self.temp, dim=1)							# 4) Compute a "soft" mask via plain softmax on clean logits
		mask = (mask_hard - mask_soft).detach() + mask_soft							# 5) Straight-through: use hard in forward, soft in backward
		return mask, topk_indices