import torch
import torch.nn as nn

from rwm.models.rwm.encoder import Encoder
from rwm.models.rwm.tokenization_head import TokenizationHead
from rwm.models.rwm.attention_scorer import AttentionScorer
from rwm.models.rwm.topk_gumbel_selector import TopKGumbelSelector
from rwm.models.rwm.patch_rnn import PatchRNN
from rwm.models.image_reconstructor.decoder import Decoder



class ImageReconstructor(nn.Module):
    def __init__(self):
        super().__init__()							# type: ignore[reportUnknownMemberType]
        self.encoder = Encoder()
        self.tokenizer = TokenizationHead()
        self.scorer = AttentionScorer()
        self.selector = TopKGumbelSelector()
        self.patch_rnn = PatchRNN()
        self.decoder = Decoder()

    
    def forward(self, img: torch.Tensor):
        feat = self.encoder(img)                    # (B, 64, 16, 16)
        tokens = self.tokenizer(feat)               # (B, N, D)

        logits = self.scorer(tokens)                # (B, N)
        mask, indices = self.selector(logits)       # mask:(B,N), indices:(B,K)

        h_spatial = self.patch_rnn(tokens, indices) # (B, hidden_dim)
        recon = self.decoder(h_spatial)             # (B, 3, 64, 64)

        return recon, mask, indices