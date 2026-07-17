"""Differentiable Top-K via Gumbel-Softmax + Straight-Through Estimator (STE).

Forward contract:
- Evaluation: hard K-hot mask, deterministic.
- Training: Gumbel-noise-tempered indices for hard mask, soft surrogate
  ``K * softmax(logits / temperature)`` for backward.

The STE mask sums to ``K`` in the soft surrogate, preserving the total
selection mass.  In eval the mask is exactly K-hot.  The mask is used
by ``SpatialAttentionHead`` to compute forward pooling weights and
backward gradients.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rwm.config.config import K


class TopKGumbelSelector(nn.Module):
    """Differentiable Top-K selector.

    Parameters
    ----------
    k:
        Number of selected tokens.
    temp:
        Temperature for Gumbel-Softmax and the soft surrogate.
    """

    def __init__(self, k: int = K, temp: float = 1.0) -> None:
        super().__init__()
        self.k = k
        self.temp = temp

    @staticmethod
    def _sample_gumbel(shape: Tuple[int, ...], device: torch.device, eps: float = 1e-20) -> Tensor:
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, logits: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (selection_mask, indices).

        Parameters
        ----------
        logits:
            ``(B, N)`` — scorer logits for all patches.

        Returns
        -------
        selection_mask:
            ``(B, N)`` — STE mask.  Forward: K-hot hard mask.
            Backward: ``K * softmax(logits / temp)`` gradient.
        indices:
            ``(B, K)`` — hard Top-K indices (used for trace/debug).
        """
        B, N = logits.shape

        # ---- Hard selection ----
        if self.training:
            noise = self._sample_gumbel((B, N), device=logits.device)
            topk_logits = (logits + noise) / self.temp
        else:
            topk_logits = logits

        indices = topk_logits.topk(self.k, dim=1).indices  # (B, K)
        hard_mask = torch.zeros_like(logits).scatter_(1, indices, 1.0)  # (B, N) K-hot

        # ---- Soft surrogate with total mass K ----
        soft = F.softmax(logits / self.temp, dim=1)  # (B, N), sums to 1
        soft_k = self.k * soft  # (B, N), sums to K

        # ---- Straight-through: forward=hard, backward=soft_k ----
        mask = (hard_mask - soft_k).detach() + soft_k
        return mask, indices
