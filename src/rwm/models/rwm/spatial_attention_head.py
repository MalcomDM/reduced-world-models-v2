import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


from rwm.config.config import TOKEN_DIM, VALUES_DIM

class SpatialAttentionHead(nn.Module):
	def __init__(self, temp: float = 1.0, dropout_p: float = 0.0):
		super().__init__()
		self.temp = temp
		self.W_v = nn.Linear(TOKEN_DIM, VALUES_DIM, bias=False)   # D -> values_dim
		self.norm = nn.LayerNorm(VALUES_DIM)
		self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()


	def forward(self, tokens: Tensor, logits: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
		B, N, D = tokens.shape
		K = indices.shape[1]

		# 1) gather de K tokens
		idx_exp 	= indices.unsqueeze(-1).expand(-1, -1, D)     # (B, K, D)
		tokkens_k   = tokens.gather(1, idx_exp)                   # (B, K, D)

		# 2) pesos: softmax sobre logits recortados a K
		log_k = logits.gather(1, indices)                     # (B, K)
		attn_k = F.softmax(log_k / self.temp, dim=1)          # (B, K)
		attn_k = self.drop(attn_k)

		# 3) values y pooling
		V_k = self.W_v(tokkens_k)                             # (B, K, values_dim)
		h   = torch.bmm(attn_k.unsqueeze(1), V_k).squeeze(1)  # (B, values_dim)

		# 4) normalización para estabilizar escala
		h = self.norm(h)                                      # (B, values_dim)
		return h, attn_k