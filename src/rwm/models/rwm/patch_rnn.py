import torch
import torch.nn as nn
from torch import Tensor

from rwm.config.config import TOKEN_DIM, PRNN_HIDDEN_DIM, K


class PatchRNN(nn.Module):
	def __init__(self) -> None:
		super().__init__()																# type: ignore[reportUnknownMemberType]
		self.gru = nn.GRUCell(input_size=TOKEN_DIM, hidden_size=PRNN_HIDDEN_DIM)
		self.hidden_dim: int = PRNN_HIDDEN_DIM
		self.k: int = K


	def forward(self, tokens:torch.Tensor, indices:torch.Tensor) -> Tensor:
		B, _N, D = tokens.shape

		# 1) Gather the K selected tokens
		idx_exp: Tensor  = indices.unsqueeze(-1).expand(-1, -1, D)   # (B, K, D)
		selected: Tensor = tokens.gather(1, idx_exp)                 # (B, K, D)

		# 2) GRUCell loop over K tokens
		h = torch.zeros(B, self.hidden_dim, device=tokens.device)
		for i in range(self.k):
			h = self.gru(selected[:, i], h)

		return h
