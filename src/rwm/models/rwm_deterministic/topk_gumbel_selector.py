import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import TOKEN_DIM, K, QUERY_DIM


def sample_gumbel(shape, eps=1e-20, device='cpu'):
    """Sample Gumel(0,1) noise"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


class TopKGumbelSelector(nn.Module):
    """
    Differentiable Top-K via Gumbel-Softmax + STE.
    forward(logits) -> mask_soft, indices
      - mask_soft: (B,N) values in [0,1], sums≈1
      - indices:  (B,K) hard indices of selected tokens
    """
    def __init__(self, temp=1.0):
        super().__init__()
        self.k = K
        self.temp = temp



    def forward(self, logits:torch.Tensor):
        """
        logits: (B, N)
        returns:
          mask_soft: (B, N) -- during training use this to multiply tokens
          indices:   (B, K) -- the hard top-K indices (for inference or gather)
        """
        B, N = logits.shape

        # 1) Sample Gumbel noise and form noisy logits
        gumbel_noise = sample_gumbel((B,N), device=logits.device)
        noisy_logits = (logits + gumbel_noise) / self.temp

        # 2) Get hard top-k indices on noisy logits
        topk = noisy_logits.topk(self.k, dim=1).indices # (B, K)

        # 3) Build hard mask
        mask_hard = torch.zeros_like(logits).scatter_(1, topk, 1.0)

        # 4) Compute a "soft" mask via plain softmax on clean logits
        mask_soft = F.softmax(logits / self.temp, dim=1)

        # 5) Straight-through: use hard in forward, soft in backward
        mask = (mask_hard - mask_soft).detach() + mask_soft
        return mask, topk