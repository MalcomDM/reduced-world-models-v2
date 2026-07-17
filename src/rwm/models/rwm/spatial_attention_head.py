"""Spatial attention head with Straight-Through Estimator (STE) selection mask.

Forward pooling equation:

    weight_i = exp(logits_i / temp) * selection_mask_i
    weight_i /= sum_j exp(logits_j / temp) * selection_mask_j

    h = sum_i weight_i * W_v(tokens_i)

``selection_mask`` is the STE mask from ``TopKGumbelSelector``:
  - Forward: K-hot hard mask (only K selected tokens contribute).
  - Backward: ``K * softmax(logits / temp)`` gradient to unselected logits.

Evaluation (hard K-hot mask):
    Only K selected tokens contribute.  The normalised weight on selected
    tokens matches previous ``indices.gather + softmax(K)`` exactly.

Training (STE mask):
    Unselected scorer logits receive finite, nonzero gradient through the
    soft surrogate component.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rwm.config.config import TOKEN_DIM, VALUES_DIM


class SpatialAttentionHead(nn.Module):
    """Spatial pooling receiving an explicit STE selection mask.

    Parameters
    ----------
    temp:
        Temperature for the weight softmax.
    dropout_p:
        Dropout probability on attention weights (training only).
    """

    def __init__(self, temp: float = 1.0, dropout_p: float = 0.0):
        super().__init__()
        self.temp = temp
        self.W_v = nn.Linear(TOKEN_DIM, VALUES_DIM, bias=False)
        self.norm = nn.LayerNorm(VALUES_DIM)
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(
        self,
        tokens: Tensor,            # (B, N, D_token)
        logits: Tensor,            # (B, N) — scorer logits (with grad)
        selection_mask: Tensor,    # (B, N) — STE mask from selector
        indices: Tensor,           # (B, K) — hard Top-K indices (debug/trace)
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Evaluation uses the original K-token gather path, so its value
        projection and pooling remain sparse. Training computes normalised
        weights over all N patches using the STE selection mask: the forward
        result uses only the hard K-hot part, while backward propagates
        through ``K * softmax``.

        Parameters
        ----------
        tokens:
            ``(B, N, D)`` — all patch tokens.
        logits:
            ``(B, N)`` — scorer logits (with gradients active).
        selection_mask:
            ``(B, N)`` — STE mask (forward: K-hot hard; backward: K*softmax).
        indices:
            ``(B, K)`` — hard Top-K indices (for backward-compatible trace).

        Returns
        -------
        h:
            ``(B, values_dim)`` — pooled spatial representation.
        attn_k:
            ``(B, K)`` — attention weights on the **selected** patches
            (for trace compatibility, extracted from the full weight tensor).
        """
        B, N, D = tokens.shape

        # Evaluation is exactly the original sparse gather path: only K value
        # vectors are projected. This preserves the intended inference cost.
        if not self.training:
            idx_exp = indices.unsqueeze(-1).expand(-1, -1, D)
            tokens_k = tokens.gather(1, idx_exp)
            logits_k = logits.gather(1, indices)
            attn_k = F.softmax(logits_k / self.temp, dim=1)
            values_k = self.W_v(tokens_k)
            h = torch.bmm(attn_k.unsqueeze(1), values_k).squeeze(1)
            return self.norm(h), attn_k

        # Training uses dense values only for the STE backward surrogate.
        # The forward selection mask is K-hot, therefore the numerical forward
        # result still pools exactly the selected K values.
        logits_scaled = logits / self.temp
        selected_max = logits_scaled.gather(1, indices).max(dim=1, keepdim=True).values
        weights_raw = (logits_scaled - selected_max).exp() * selection_mask
        weights = weights_raw / weights_raw.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # (B, N)
        weights = self.drop(weights)

        # ---- Pool ----
        V = self.W_v(tokens)  # (B, N, values_dim)
        h = torch.bmm(weights.unsqueeze(1), V).squeeze(1)  # (B, values_dim)
        h = self.norm(h)

        # ---- Extract K-sized attention weights for trace compat ----
        attn_k = weights.gather(1, indices)  # (B, K)

        return h, attn_k
