import torch
import torch.nn as nn

from rwm.config.config import ACTION_DIM, OBSERVATIONAL_DROPOUT
from rwm.models.rwm_deterministic.encoder import Encoder
from rwm.models.rwm_deterministic.tokenization_head import TokenizationHead
from rwm.models.rwm_deterministic.attention_scorer import AttentionScorer
from rwm.models.rwm_deterministic.topk_gumbel_selector import TopKGumbelSelector
from rwm.models.rwm_deterministic.patch_rnn import PatchRNN
from rwm.models.rwm_deterministic.world_rnn import WorldRNN

class ReducedWorldModel(nn.Module):

    def __init__(self, action_dim: int = ACTION_DIM, dropout_prob: float = OBSERVATIONAL_DROPOUT):
        super().__init__()							# type: ignore[reportUnknownMemberType]

        self.encoder = Encoder()
        self.tokenizer = TokenizationHead()
        self.scorer = AttentionScorer()
        self.selector = TopKGumbelSelector()
        self.patch_rnn = PatchRNN()
        self.world_rnn = WorldRNN(action_dim=action_dim, dropout_prob=dropout_prob)

    def forward(
        self,
        img: torch.Tensor,                          # (B, 3, 64, 64)        imagen actual del entorno
        a_prev: torch.Tensor,                       # (B, action_dim)       acción tomada en t-1 (batch x action_dim)
        h_prev: torch.Tensor,                       # (B, WRNN_HIDDEN_DIM)  estado oculto de WorldRNN en t-1 (batch x WRNN_HIDDEN_DIM)
        c_prev: torch.Tensor,                       # Añadido
        force_keep_input: bool = False              # si True → no aplicamos observational dropout en este paso
    ):        
        feat = self.encoder(img)                    # (B, 64, 16, 16)
        tokens = self.tokenizer(feat)               # (B, N, D), donde N = (H'·W'), D = TOKEN_DIM
        logits = self.scorer(tokens)                # (B, N)
        mask, indices = self.selector(logits)       # mask:(B,N), indices:(B,K)
        h_spatial = self.patch_rnn(tokens, indices) # (B, PRNN_HIDDEN_DIM)

        h_new, _, r_pred = self.world_rnn(			# c_new as _
            h_prev=h_prev,
            c_prev=c_prev,
            x_spatial=h_spatial,
            a_prev=a_prev,
            force_keep_input=force_keep_input
        )

        return h_new, c_prev, r_pred, mask, indices