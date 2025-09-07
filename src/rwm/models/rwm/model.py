import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from rwm.config.config import ACTION_DIM, OBSERVATIONAL_DROPOUT
from rwm.models.rwm.encoder import Encoder
from rwm.models.rwm.tokenization_head import TokenizationHead
from rwm.models.rwm.attention_scorer import AttentionScorer
from rwm.models.rwm.topk_gumbel_selector import TopKGumbelSelector
from rwm.models.rwm.spatial_attention_head import SpatialAttentionHead
from rwm.models.rwm.casusal_transformer import CausalTransformer



class ReducedWorldModel(nn.Module):

	def __init__(self, action_dim: int = ACTION_DIM, dropout_prob: float = OBSERVATIONAL_DROPOUT):
		super().__init__()							# type: ignore[reportUnknownMemberType]

		self.encoder = Encoder()
		self.tokenizer = TokenizationHead()
		self.scorer = AttentionScorer()
		self.selector = TopKGumbelSelector()
		self.spatial_hd = SpatialAttentionHead()
		self.world_hd = CausalTransformer()

	def forward(
		self,
		img: torch.Tensor,                          # (B, 3, 64, 64)        imagen actual del entorno
		a_prev: torch.Tensor,                       # (B, action_dim)       acción tomada en t-1 (batch x action_dim)
		h_prev: torch.Tensor,                       # (B, WRNN_HIDDEN_DIM)  estado oculto de WorldRNN en t-1 (batch x WRNN_HIDDEN_DIM)
		c_prev: torch.Tensor,                       # Añadido
		force_keep_input: bool = False              # si True → no aplicamos observational dropout en este paso
	) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
		feat = self.encoder(img)                    # (B, 64, 16, 16)
		tokens = self.tokenizer(feat)               # (B, N, D), donde N = (H'·W'), D = TOKEN_DIM
		logits = self.scorer(tokens)                # (B, N)
		mask, indices = self.selector(logits)       # mask:(B,N), indices:(B,K)
		h_esp, w = self.spatial_hd(tokens, mask)	# (B, PRNN_HIDDEN_DIM)

		h_new, _, r_pred = self.world_hd(			# c_new as _
			h_prev=h_prev,
			c_prev=c_prev,
			x_spatial=h_esp,
			a_prev=a_prev,
			force_keep_input=force_keep_input
		)

		return h_new, c_prev, r_pred, mask, indices
	

	def generate_spatial_rep(self, img_t: Tensor) -> Tensor:
		feat = self.encoder(img_t)
		tokens = self.tokenizer(feat)
		logits = self.scorer(tokens)
		_, idx = self.selector(logits)
		h_spatial: Tensor = self.spatial_hd(tokens, idx)

		return h_spatial