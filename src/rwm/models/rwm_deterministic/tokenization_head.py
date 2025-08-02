import torch
import torch.nn as nn

from app.config import TKN_IN_CHANNELS, NUM_PATCH_DIM, PATCH_SIZE, PATCH_STRIDE, PATCH_PADDING, TOKEN_DIM


class TokenizationHead(nn.Module):

    @staticmethod
    def _positional_encoding(NDim, D) -> torch.Tensor:
        assert D % 2 == 0 # D must be even; we’ll split half for x, half for y
        d = D // 2
        pe = torch.zeros(NDim*NDim, D)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))

        pos_x = torch.arange(NDim, dtype=torch.float32).unsqueeze(1)
        sincos_x = torch.cat([torch.sin(pos_x * inv_freq),
                              torch.cos(pos_x * inv_freq)], dim=1)
        pos_y = torch.arange(NDim, dtype=torch.float32).unsqueeze(1)
        sincos_y = torch.cat([torch.sin(pos_y * inv_freq),
                              torch.cos(pos_y * inv_freq)], dim=1)
        idx = 0
        for y in range(NDim):
            for x in range(NDim):
                pe[idx, :d] = sincos_x[x]
                pe[idx, d:] = sincos_y[y]
                idx += 1
        return pe.unsqueeze(0) # shape (1, H*W, D)
    
    # Build one global pos‐embed buffer for all instances
    pos_embed = _positional_encoding(NUM_PATCH_DIM, TOKEN_DIM)


    def __init__(self):
        super().__init__()
        P, S, pad = PATCH_SIZE, PATCH_STRIDE, PATCH_PADDING

        self.unfold = nn.Unfold(kernel_size=P, stride=S, padding=pad)
        self.projection = nn.Linear(TKN_IN_CHANNELS * P * P, TOKEN_DIM)

        # Register the shared positional-encoding buffer
        self.register_buffer('pos_embed_buffer', TokenizationHead.pos_embed)


    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W)
        patches = self.unfold(feature_map)                  # (B, C*P*P, N)
        patches: torch.Tensor  = patches.permute(0, 2, 1)   # (B, N, C*P*P)
        tokens = self.projection(patches)                   # (B, N, token_dim)
        tokens = tokens + self.pos_embed_buffer             # broadcast over batch
        return tokens

