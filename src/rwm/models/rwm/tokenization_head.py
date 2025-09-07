import torch
import torch.nn as nn
from torch import Tensor
from typing import ClassVar, Tuple

from rwm.config.config import (
    PATCH_SIZE, PATCH_STRIDE, PATCH_PADDING,
	TKN_IN_CHANNELS, TOKEN_DIM, PATCHES_PER_SIDE, 
)



def _build_positional_encoding(n_dim: int, d_model: int) -> torch.Tensor:
	""" Build a (1, n_dim*n_dim, d_model) sinusoidal positional encoding. """
	assert d_model % 2 == 0 # TOKEN_DIM must be even; will be splited half for x, half for y
	half = d_model // 2
	pe = torch.zeros(n_dim*n_dim, d_model)
	inv_freq = 1.0 / (10000 ** (torch.arange(0, half, 2, dtype=torch.float32) / half))

	pos = torch.arange(n_dim, dtype=torch.float32).unsqueeze(1)  							# (n_dim,1)
	sincos_x = torch.cat([torch.sin(pos * inv_freq), torch.cos(pos * inv_freq)], dim=1)		# (n_dim, half)
	sincos_y = sincos_x																		# same frequencies for Y
	
	idx = 0
	for y in range(n_dim):
		for x in range(n_dim):
			pe[idx, :half] = sincos_x[x]
			pe[idx, half:] = sincos_y[y]
			idx += 1
	return pe.unsqueeze(0) # shape (1, H*W, D)


_POS_EMBED: Tensor = _build_positional_encoding(PATCHES_PER_SIDE, TOKEN_DIM)


class TokenizationHead(nn.Module):
	"""
    A variational tokenization head inspired by VAEs.
    Maps patches to a distribution (mean, log_variance) and samples from it.
	"""
	pos_embed_buffer: ClassVar[Tensor] 


	def __init__(self) -> None:
		super().__init__()				# type: ignore[reportUnknownMemberType]
		P, S, pad = PATCH_SIZE, PATCH_STRIDE, PATCH_PADDING

		self.unfold = nn.Unfold(kernel_size=P, stride=S, padding=pad)		 # Unfold layer: (B, C, H, W) -> (B, C*P*P, N)
		self.projection = nn.Linear(TKN_IN_CHANNELS * P * P, TOKEN_DIM * 2 ) # Linear projection of each flattened patch to mean and log_var

		# Register the shared positional-encoding buffer
		self.register_buffer('pos_embed_buffer', _POS_EMBED)


	def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		patches = self.unfold(feature_map).permute(0, 2, 1)					# (B, C, H, W) -> (B, C*P*P, N) -> (B, N, C*P*P)

		# 1) Project to mean and log_variance
		projected_patches = self.projection(patches)						# (B, N, TOKEN_DIM * 2)
		mean, log_var = torch.chunk(projected_patches, 2, dim=-1)			# (B, N, TOKEN_DIM ) * 2

		# 2) Reparameterization trick
		std = torch.exp(0.5 * log_var)
		eps = torch.randn_like(std)
		tokens = mean + eps * std

		# 3) Add positional embeddings
		tokens = tokens + self.pos_embed_buffer
		return tokens, mean, log_var

