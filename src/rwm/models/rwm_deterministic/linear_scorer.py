import torch
import torch.nn as nn

from app.config import TOKEN_DIM, PRNN_HIDDEN_DIM


class PatchScorer(nn.Module):
    """Turns each D-d token into a scalar score."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TOKEN_DIM, PRNN_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(PRNN_HIDDEN_DIM, 1),
        )

    
    def forward(self, tokens):  # tokens: (B, N, D)
        # returns (B, N) logits
        return self.net(tokens).squeeze(-1)