import math
import torch
import torch.nn as nn

from rwm.config.config import TOKEN_DIM, QUERY_DIM


class AttentionScorer(nn.Module):
    """
    Computes a single “global” attention score per patch by
    cross‐attending your RNN’s previous hidden state (or a
    learned query) to all tokens.
    """
    def __init__(self):
        super().__init__()												# type: ignore[reportUnknownMemberType]
        self.to_k = nn.Linear(TOKEN_DIM, QUERY_DIM, bias=False)
        self.query = nn.Parameter(torch.rand(1, QUERY_DIM))


    def forward(self, tokens:torch.Tensor) -> torch.Tensor:  # tokens: (B, N, D)
        B, _, _ = tokens.size()
        Q = self.query.unsqueeze(0).expand(B, 1, -1)    # (B, 1, dk)
        K:torch.Tensor = self.to_k(tokens)              # (B, N, dk)

        # Scaled dot-product
        scores = (Q @ K.transpose(-2, -1)).squeeze(1)   # (B, N)
        logits = scores / math.sqrt(K.size(-1))
        
        return logits 