
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rwm.config.config import VALUES_DIM, ACTION_DIM, WORLD_STATE_DIM, SEQ_LEN

class CausalTransformer(nn.Module):
    def __init__( self, ffn_mult: int = 2, dropout: float = 0.1 ):
        super().__init__()
        self.d_model = WORLD_STATE_DIM
        self.max_seq_len = SEQ_LEN
        input_dim = VALUES_DIM + ACTION_DIM

        self.input_proj = nn.Linear(input_dim, WORLD_STATE_DIM)					# Proyección a d_model
        self.pos_emb = nn.Embedding(SEQ_LEN, WORLD_STATE_DIM)					# Embeddings posicionales aprendidos (simples y efectivos en T corto)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=WORLD_STATE_DIM,
            nhead=1,
            dim_feedforward=ffn_mult * WORLD_STATE_DIM,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.reward_head = nn.Linear(WORLD_STATE_DIM, 1)

    @staticmethod
    def _causal_mask(T: int, device) -> Tensor:
        m = torch.full((T, T), float('-inf'), device=device)
        return torch.triu(m, diagonal=1)  # prohibe atender al futuro

    @staticmethod
    def _kpm_from_lengths(lengths: Tensor, T: int) -> Tensor:
        # True = mask (padding)
        r = torch.arange(T, device=lengths.device).unsqueeze(0)
        return r >= lengths.unsqueeze(1)

    def forward(self, history: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        history: (B, T, input_dim)  con padding a la derecha si aplica
        lengths: (B,) longitudes reales opcionales
        """
        B, T, D = history.shape
        assert T <= self.max_seq_len, "Aumenta max_seq_len o recorta T."

        x = self.input_proj(history)                        # (B, T, d_model)
        pos = torch.arange(T, device=history.device).unsqueeze(0)  # (1, T)
        x = x + self.pos_emb(pos)                           # (B, T, d_model)

        mask = self._causal_mask(T, history.device)         # (T, T)
        kpm = self._kpm_from_lengths(lengths, T) if lengths is not None else None  # (B, T)

        out = self.encoder(x, mask=mask, src_key_padding_mask=kpm)   # (B, T, d_model)

        # último token válido por batch
        if lengths is None:
            world_state = out[:, -1, :]
        else:
            idx = (lengths - 1).clamp_min(0)
            world_state = out[torch.arange(B, device=out.device), idx, :]

        r_pred = self.reward_head(world_state)              # (B, 1)
        return world_state, r_pred