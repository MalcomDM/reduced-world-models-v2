"""End-to-end reduced world model: perception → temporal → controller trunk.

Timing contract (approved):

    belief b_t = Transformer(obs[t], action[t-1], history)
    shared_repr = ControllerTrunk.encode(b_t)
    Actor(shared_repr)            → action[t]          (Stage 4)
    Critic(shared_repr)           → V(b_t)             (Stage 4)
    ControllerTrunk.predict_reward(shared_repr, action[t])
                                  → reward[t] (= r_{t+1})

Reference: ``docs/transition_contract.md``
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from rwm.config.config import (
    ACTION_DIM,
    OBSERVATIONAL_DROPOUT,
    SEQ_LEN,
    VALUES_DIM,
)
from rwm.models.rwm.encoder import Encoder
from rwm.models.rwm.observational_dropout import ObservationalDropout
from rwm.models.rwm.tokenization_head import TokenizationHead
from rwm.models.rwm.attention_scorer import AttentionScorer
from rwm.models.rwm.topk_gumbel_selector import TopKGumbelSelector
from rwm.models.rwm.spatial_attention_head import SpatialAttentionHead
from rwm.models.rwm.causal_transformer import CausalTransformer
from rwm.models.controller_trunk import ControllerTrunk
from rwm.types import WorldModelOutput
from rwm.utils.history_buffer import HistoryBuffer


def _perceive_frame(
    encoder: nn.Module,
    tokenizer: nn.Module,
    scorer: nn.Module,
    selector: nn.Module,
    spatial_hd: nn.Module,
    obs_drop: nn.Module,
    img: torch.Tensor,
    force_keep: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           Optional[torch.Tensor], Optional[torch.Tensor]]:
    feat = encoder(img)
    tok_out = tokenizer(feat)
    tokens = tok_out if isinstance(tok_out, torch.Tensor) else tok_out[0]
    tok_mu: Optional[torch.Tensor] = None
    tok_logvar: Optional[torch.Tensor] = None
    if isinstance(tok_out, tuple) and len(tok_out) >= 3:
        _, tok_mu, tok_logvar = tok_out
    logits = scorer(tokens)
    selection_mask, indices = selector(logits)
    h_spatial, _attn_k = spatial_hd(tokens, logits, selection_mask, indices)
    h_spatial = obs_drop(h_spatial, force_keep=force_keep)
    return h_spatial, selection_mask, indices, tok_mu, tok_logvar


class ReducedWorldModel(nn.Module):
    """End-to-end reduced world model.

    Timing contract::

        token[t] = cat(spatial_rep(obs[t]), action[t-1])
        Transformer processes tokens[0..t] → belief_t   (action-free)
        ControllerTrunk.encode(belief_t)   → shared_repr (actor-ready)
        ControllerTrunk.predict_reward(shared_repr, action[t])
                                          → reward[t]
    """

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        dropout_prob: float = OBSERVATIONAL_DROPOUT,
    ):
        super().__init__()
        self.encoder = Encoder()
        self.tokenizer = TokenizationHead()
        self.scorer = AttentionScorer()
        self.selector = TopKGumbelSelector()
        self.spatial_hd = SpatialAttentionHead()
        self.obs_drop = ObservationalDropout(p=dropout_prob, mode="zero")
        self.world_hd = CausalTransformer()
        self.controller = ControllerTrunk(action_dim=action_dim)

    # ------------------------------------------------------------------
    # Incremental inference
    # ------------------------------------------------------------------

    def forward(
        self,
        img: torch.Tensor,                     # (B, 3, H, W)
        prev_action: torch.Tensor,             # (B, A) action[t-1]; zeros at t=0
        current_action: torch.Tensor,          # (B, A) action[t] for reward head
        history: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        force_keep_input: bool = False,
    ) -> WorldModelOutput:
        """Incremental forward pass.

        Builds a belief from ``obs[t]`` and ``prev_action`` (action[t-1]),
        then predicts reward using that belief and ``current_action``
        (action[t]).
        """
        h_spatial, mask_soft, indices, tok_mu, tok_logvar = _perceive_frame(
            self.encoder, self.tokenizer, self.scorer,
            self.selector, self.spatial_hd, self.obs_drop,
            img, force_keep_input,
        )

        # Token uses PREVIOUS action (action[t-1]).
        token_t = torch.cat([h_spatial, prev_action], dim=-1).unsqueeze(1)

        if history is not None:
            buf = HistoryBuffer.from_history(
                max_seq_len=SEQ_LEN,
                input_dim=VALUES_DIM + ACTION_DIM,
                history=history, lengths=lengths,
            )
        else:
            buf = HistoryBuffer(
                max_seq_len=SEQ_LEN,
                input_dim=VALUES_DIM + ACTION_DIM,
                device=img.device,
            )
        hist_seq, hist_len = buf.append(token_t)

        # Belief from Transformer (pre-action).
        belief = self.world_hd(hist_seq, lengths=hist_len)

        # Reward prediction uses CURRENT action (action[t]).
        _shared_repr, reward_pred = self.controller(belief, current_action)

        return WorldModelOutput(
            world_state=belief,
            reward_pred=reward_pred,
            mask_soft=mask_soft,
            indices=indices,
            history=hist_seq,
            lengths=hist_len,
            tok_mu=tok_mu,
            tok_logvar=tok_logvar,
        )

    # ------------------------------------------------------------------
    # Full-sequence batched training
    # ------------------------------------------------------------------

    def forward_sequence(
        self,
        obs: torch.Tensor,          # (B, T, C, H, W)
        prev_actions: torch.Tensor, # (B, T, A) prev_actions[:,0]=zeros, prev_actions[:,t]=action[t-1]
        current_actions: torch.Tensor,  # (B, T, A) action[t] for reward head at each position
        force_keep_input: bool = False,
    ) -> WorldModelOutput:
        """Full-sequence forward pass — vectorised over ``B*T`` frames.

        Perception (encoder → tokenizer → scorer → selector → spatial head)
        is applied once to all ``B*T`` frames reshaped into a single batch.
        Controller reward prediction is similarly vectorised.
        """
        B, T = obs.shape[0], obs.shape[1]
        input_dim = VALUES_DIM + ACTION_DIM
        device = obs.device

        # ----- Vectorised perception over B*T frames -----
        frames = obs.reshape(B * T, *obs.shape[2:])  # (B*T, C, H, W)

        feat = self.encoder(frames)                   # (B*T, C', H', W')
        tok_out = self.tokenizer(feat)

        if isinstance(tok_out, torch.Tensor):
            tokens = tok_out
            tok_mu = None
            tok_logvar = None
        else:
            tokens, tok_mu, tok_logvar = tok_out

        logits = self.scorer(tokens)                  # (B*T, N)
        selection_mask, indices = self.selector(logits)  # (B*T, N), (B*T, K)
        h_spatial, attn_k = self.spatial_hd(tokens, logits, selection_mask, indices)  # (B*T, V)
        h_spatial = self.obs_drop(h_spatial, force_keep=force_keep_input)

        # ----- Build Transformer tokens -----
        h_spatial_bt = h_spatial.view(B, T, -1)       # (B, T, values_dim)
        all_tokens = torch.cat([h_spatial_bt, prev_actions], dim=-1)  # (B, T, input_dim)

        # Aggregated per-frame outputs.
        selection_mask_bt = selection_mask.view(B, T, -1)   # (B, T, N)
        indices_bt = indices.view(B, T, -1)                  # (B, T, K)
        if tok_mu is not None:
            tok_mu_bt = tok_mu.view(B, T, *tok_mu.shape[1:])     # (B, T, P, D)
            tok_logvar_bt = tok_logvar.view(B, T, *tok_logvar.shape[1:])
        else:
            tok_mu_bt = None
            tok_logvar_bt = None

        # ----- Single Transformer pass -----
        lengths = torch.full((B,), T, device=device, dtype=torch.long)
        all_out = self.world_hd(all_tokens, lengths=lengths, return_all=True)  # (B, T, D)

        # ----- Vectorised controller reward over B*T positions -----
        out_bt = all_out.reshape(B * T, -1)            # (B*T, D)
        curr_act_bt = current_actions.reshape(B * T, -1)  # (B*T, A)
        _shared_bt, r_pred_bt = self.controller(out_bt, curr_act_bt)
        reward_pred_seq = r_pred_bt.view(B, T)          # (B, T)

        # Last position for API compatibility; reuse the vectorised prediction
        # so ``reward_pred`` is exactly the final sequence prediction.
        belief_last = all_out[:, -1]
        rp_last = reward_pred_seq[:, -1:].contiguous()

        return WorldModelOutput(
            world_state=belief_last,
            reward_pred=rp_last,
            mask_soft=selection_mask_bt[:, -1],      # last frame's mask
            indices=indices_bt[:, -1],                # last frame's indices
            history=all_tokens,
            lengths=lengths,
            tok_mu=tok_mu_bt,
            tok_logvar=tok_logvar_bt,
            reward_pred_seq=reward_pred_seq,
        )

    # ------------------------------------------------------------------
    # Spatial representation (inference)
    # ------------------------------------------------------------------

    def generate_spatial_rep(self, img_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h_spatial, _mask, _idx, _mu, _lv = _perceive_frame(
                self.encoder, self.tokenizer, self.scorer,
                self.selector, self.spatial_hd, self.obs_drop,
                img_t, force_keep=True,
            )
            return h_spatial
