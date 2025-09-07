import torch
import torch.nn as nn
from torch import Tensor

from src.rwm.config.config import TOKEN_DIM


class ObservationalDropout(nn.Module):
	def __init__(self, p: float = 0.8, mode: str = "zero"):
		super().__init__()
		self.p = p
		self.mode = mode
		if mode == "token":
			self.mask_token = nn.Parameter(torch.zeros(1, TOKEN_DIM))
			nn.init.normal_(self.mask_token, std=0.02)


	def forward(self, x: Tensor, force_keep: bool = False) -> Tensor:
		# x: (B,D)
		if force_keep or (not self.training) or self.p <= 0:
			return x
		B, D = x.shape
		keep = (torch.rand(B, 1, device=x.device) > self.p).float()  # 1=mantener, 0=cegar
		if self.mode == "zero":
			return x * keep
		else:
			return x * keep + (1 - keep) * self.mask_token.expand(B, D)