"""End-to-end reduced world model: perception → temporal → controller trunk.

Timing contract (approved):

    belief b_t = Transformer(obs[t], action[t-1], history)
    shared_repr = ControllerTrunk.encode(b_t)
    Actor(shared_repr)            → action[t]          (Stage 4)
    Critic(shared_repr)           → V(b_t)             (Stage 4)
    ControllerTrunk.predict_reward(shared_repr, action[t])
                                  → reward[t] (= r_{t+1})

SRU alternative (S1):

    z_t = MinimalSRUTemporal.step(x_t, z_{t-1})
    shared_repr = ControllerTrunk.encode(z_t)
    ...

Reference: ``docs/contracts/transition_contract.md``
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from rwm.config.config import (
    ACTION_DIM,
    OBSERVATIONAL_DROPOUT,
    SEQ_LEN,
    VALUES_DIM,
    WORLD_STATE_DIM,
    NUM_PATCHES,
)
from rwm.config.experiment_config import TemporalConfig
from rwm.models.rwm.encoder import Encoder
from rwm.models.rwm.observational_dropout import ObservationalDropout
from rwm.models.rwm.tokenization_head import TokenizationHead
from rwm.models.rwm.attention_scorer import AttentionScorer
from rwm.models.rwm.topk_gumbel_selector import TopKGumbelSelector
from rwm.models.rwm.spatial_attention_head import SpatialAttentionHead
from rwm.models.rwm.causal_transformer import CausalTransformer
from rwm.models.rwm.minimal_sru_temporal import MinimalSRUTemporal
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

    Supports two temporal backends selected by ``temporal_config``:

    * ``"causal_transformer"`` (default): bounded-context CausalTransformer.
    * ``"minimal_sru"``: compact single-gate recurrent cell.

    Timing contract (causal)::

        token[t] = cat(spatial_rep(obs[t]), action[t-1])
        Transformer processes tokens[0..t] → belief_t

    Timing contract (SRU)::

        x_t = cat(spatial_rep * keep_bit, action[t-1], keep_bit)
        z_t = MinimalSRUTemporal.step(x_t, z_{t-1})
    """

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        dropout_prob: float = OBSERVATIONAL_DROPOUT,
        reward_head_kind: str = "linear",
        reward_head_hidden_dim: int = 32,
        selection_mode: str = "learned",
        selection_k: int = 8,
        selection_seed: int = 0,
        tokenizer_eval_mode: str = "sample",
        temporal_config: Optional[TemporalConfig] = None,
    ):
        super().__init__()
        assert tokenizer_eval_mode in ("sample", "mean")
        self._reward_head_kind = reward_head_kind
        self._reward_head_hidden_dim = reward_head_hidden_dim
        self._selection_mode = selection_mode
        self._selection_k = selection_k
        self._tokenizer_eval_mode = tokenizer_eval_mode

        if temporal_config is None:
            temporal_config = TemporalConfig()
        self._temporal_config = temporal_config
        self._temporal_backend = temporal_config.backend

        self.encoder = Encoder()
        self.tokenizer = TokenizationHead(eval_mode=tokenizer_eval_mode)
        self.scorer = AttentionScorer()
        self.selector = TopKGumbelSelector(
            k=selection_k, selection_mode=selection_mode,
            selection_seed=selection_seed,
        )
        self.spatial_hd = SpatialAttentionHead()
        self.obs_drop = ObservationalDropout(p=dropout_prob, mode="zero")

        if self._temporal_backend == "causal_transformer":
            self.world_hd = CausalTransformer()
        elif self._temporal_backend == "minimal_sru":
            self.world_hd = MinimalSRUTemporal(
                input_dim=VALUES_DIM + ACTION_DIM + 1,  # spatial + action + visibility bit
                state_dim=WORLD_STATE_DIM,
                carry_bias_init=temporal_config.sru_carry_bias_init,
            )
        else:
            raise ValueError(f"Unknown temporal backend: {self._temporal_backend!r}")

        self.controller = ControllerTrunk(
            action_dim=action_dim,
            reward_head_kind=reward_head_kind,
            reward_head_hidden_dim=reward_head_hidden_dim,
        )

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
        observation_keep: Optional[bool] = None,  # per-step; False=masked, True=visible, None=visible
        temporal_state: Optional[torch.Tensor] = None,  # (B, D) SRU recurrent state
        valid_step: Optional[torch.Tensor] = None,  # (B,) bool; SRU only; True=normal, False=padding
    ) -> WorldModelOutput:
        """Incremental forward pass.

        Builds a belief from ``obs[t]`` and ``prev_action`` (action[t-1]),
        then predicts reward using that belief and ``current_action``
        (action[t]).

        When ``observation_keep`` is False, the spatial representation is
        replaced with zeros (the image is still processed for diagnostics).
        """
        h_spatial, mask_soft, indices, tok_mu, tok_logvar = _perceive_frame(
            self.encoder, self.tokenizer, self.scorer,
            self.selector, self.spatial_hd, self.obs_drop,
            img, force_keep_input,
        )

        # Apply observation mask (per-step).
        if observation_keep is False:
            h_spatial = torch.zeros_like(h_spatial)

        if self._temporal_backend == "minimal_sru":
            # ---- SRU incremental path ----
            B_sru = img.shape[0]
            keep_bit = torch.zeros(B_sru, 1, device=img.device, dtype=h_spatial.dtype)
            if observation_keep is not False:
                keep_bit.fill_(1.0)

            x_t = torch.cat([h_spatial, prev_action, keep_bit], dim=-1)  # (B, 36)

            if temporal_state is None:
                if history is not None:
                    raise ValueError(
                        "SRU cannot use causal history as recurrent state. "
                        "Pass temporal_state (from the previous output) instead."
                    )
                z_prev = None  # zeros
            else:
                z_prev = temporal_state

            z_t = self.world_hd.step(x_t, z_prev=z_prev, valid_step=valid_step)

            # Placeholder history for backward compat (not consumed by SRU callers).
            compat_token = torch.cat([h_spatial, prev_action], dim=-1).unsqueeze(1)

            # Reward prediction uses CURRENT action (action[t]).
            _shared_repr, reward_pred = self.controller(z_t, current_action)

            return WorldModelOutput(
                world_state=z_t,
                reward_pred=reward_pred,
                mask_soft=mask_soft,
                indices=indices,
                history=compat_token,
                lengths=torch.full((B_sru,), 1, device=img.device, dtype=torch.long),
                tok_mu=tok_mu,
                tok_logvar=tok_logvar,
                temporal_state=z_t,
            )

        # ---- Causal Transformer incremental path (unchanged) ----
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
            temporal_state=None,
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
        observation_keep: Optional[torch.Tensor] = None,  # (B, T) bool; True=visible, False=masked
        temporal_state: Optional[torch.Tensor] = None,  # (B, D) SRU initial state
        valid_step: Optional[torch.Tensor] = None,  # (B, T) bool; SRU only; True=normal, False=padding
        observation_dropout_execution: Optional[str] = None,  # "post_perception" | "pre_perception_skip"
    ) -> WorldModelOutput:
        """Full-sequence forward pass — vectorised over ``B*T`` frames.

        Perception (encoder → tokenizer → scorer → selector → spatial head)
        is applied once to all ``B*T`` frames reshaped into a single batch.
        Controller reward prediction is similarly vectorised.

        When ``observation_keep`` is provided, spatial representations at
        masked (False) positions are handled according to
        ``observation_dropout_execution``:

        * ``"post_perception"`` (default): perception still runs on all frames;
          the spatial output is zeroed after perception.  Diagnostic outputs
          remain valid for all frames.
        * ``"pre_perception_skip"``: masked frames bypass perception entirely.
          A zero spatial representation and sentinel diagnostics (mask_soft=0,
          indices=-1, tok_mu/logvar=None) are injected directly.
        """
        B, T = obs.shape[0], obs.shape[1]
        input_dim = VALUES_DIM + ACTION_DIM
        device = obs.device
        exec_policy = observation_dropout_execution or "post_perception"

        # ---- Pre-perception skip path ----
        if exec_policy == "pre_perception_skip" and observation_keep is not None:
            n_vis = observation_keep.sum().item()
            if n_vis < B * T:
                # Split visible and masked positions.
                keep_float = observation_keep.unsqueeze(-1).float()  # (B, T, 1)

                # Visible frames: run perception only on those positions.
                visible_flat = obs[observation_keep]  # (n_vis, C, H, W)
                if n_vis > 0:
                    v_feat = self.encoder(visible_flat)
                    v_tok = self.tokenizer(v_feat)
                    if isinstance(v_tok, torch.Tensor):
                        v_tokens, v_mu, v_lv = v_tok, None, None
                    else:
                        v_tokens, v_mu, v_lv = v_tok
                    v_logits = self.scorer(v_tokens)
                    v_sel_mask, v_idx = self.selector(v_logits)
                    v_h, _v_attn = self.spatial_hd(v_tokens, v_logits, v_sel_mask, v_idx)
                    v_h = self.obs_drop(v_h, force_keep=force_keep_input)
                else:
                    v_h = torch.empty(0, VALUES_DIM, device=device)

                # Build full outputs with sentinel for masked positions.
                zeros_bt = torch.zeros(B, T, VALUES_DIM, device=device)
                sel_mask_full = torch.zeros(B, T, NUM_PATCHES, device=device)
                # ``selection_k`` is configurable (K-ablation and checkpoints
                # need not use the default K=8), so diagnostics must follow the
                # active selector rather than hard-code its historical default.
                idx_full = torch.full(
                    (B, T, self.selector.k), -1, device=device, dtype=torch.long,
                )
                tok_mu_full = None
                tok_logvar_full = None
                if n_vis > 0:
                    zeros_bt[observation_keep] = v_h
                    # Selection mask and indices from visible path.
                    v_sm_flat = v_sel_mask  # (n_vis, N)
                    v_idx_flat = v_idx      # (n_vis, K)
                    # Only fill first position if visible is contiguous — easier:
                    sel_mask_full[observation_keep] = v_sm_flat
                    idx_full[observation_keep] = v_idx_flat
                    if v_mu is not None:
                        tok_mu_full = torch.zeros(B, T, *v_mu.shape[1:], device=device)
                        tok_logvar_full = torch.zeros(B, T, *v_lv.shape[1:], device=device)
                        tok_mu_full[observation_keep] = v_mu
                        tok_logvar_full[observation_keep] = v_lv

                h_spatial_bt = zeros_bt
                selection_mask_bt = sel_mask_full
                indices_bt = idx_full
                tok_mu_bt = tok_mu_full
                tok_logvar_bt = tok_logvar_full

                if self._temporal_backend == "minimal_sru":
                    x_sru = torch.cat([h_spatial_bt, prev_actions, keep_float], dim=-1)
                    z_all, z_last = self.world_hd.forward_sequence(
                        x_sru, initial_state=temporal_state,
                        valid_step=valid_step, return_all=True,
                    )
                    out_bt = z_all.reshape(B * T, -1)
                    curr_act_bt = current_actions.reshape(B * T, -1)
                    _shared_bt, r_pred_bt = self.controller(out_bt, curr_act_bt)
                    reward_pred_seq = r_pred_bt.view(B, T)
                    rp_last = reward_pred_seq[:, -1:].contiguous()
                    compat_history = torch.cat([h_spatial_bt, prev_actions], dim=-1)
                    return WorldModelOutput(
                        world_state=z_last, reward_pred=rp_last,
                        mask_soft=selection_mask_bt[:, -1], indices=indices_bt[:, -1],
                        history=compat_history,
                        lengths=torch.full((B,), T, device=device, dtype=torch.long),
                        tok_mu=tok_mu_bt, tok_logvar=tok_logvar_bt,
                        reward_pred_seq=reward_pred_seq, temporal_state=z_last,
                    )
                # Causal path for pre_perception_skip (rare but supported).
                all_tokens = torch.cat([h_spatial_bt, prev_actions], dim=-1)
                lengths = torch.full((B,), T, device=device, dtype=torch.long)
                all_out = self.world_hd(all_tokens, lengths=lengths, return_all=True)
                out_bt = all_out.reshape(B * T, -1)
                curr_act_bt = current_actions.reshape(B * T, -1)
                _shared_bt, r_pred_bt = self.controller(out_bt, curr_act_bt)
                reward_pred_seq = r_pred_bt.view(B, T)
                belief_last = all_out[:, -1]
                rp_last = reward_pred_seq[:, -1:].contiguous()
                return WorldModelOutput(
                    world_state=belief_last, reward_pred=rp_last,
                    mask_soft=selection_mask_bt[:, -1], indices=indices_bt[:, -1],
                    history=all_tokens, lengths=lengths,
                    tok_mu=tok_mu_bt, tok_logvar=tok_logvar_bt,
                    reward_pred_seq=reward_pred_seq, temporal_state=None,
                )

        # ----- Vectorised perception over B*T frames (default / legacy path) -----
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

        # ----- Observation visibility mask -----
        keep_float = torch.ones(B, T, 1, device=device, dtype=torch.float32)
        if observation_keep is not None:
            keep_float = observation_keep.unsqueeze(-1).float()  # (B, T, 1)
            h_spatial_bt = h_spatial.view(B, T, -1) * keep_float
        else:
            h_spatial_bt = h_spatial.view(B, T, -1)       # (B, T, values_dim)

        # Aggregated per-frame outputs.
        selection_mask_bt = selection_mask.view(B, T, -1)   # (B, T, N)
        indices_bt = indices.view(B, T, -1)                  # (B, T, K)
        if tok_mu is not None:
            tok_mu_bt = tok_mu.view(B, T, *tok_mu.shape[1:])     # (B, T, P, D)
            tok_logvar_bt = tok_logvar.view(B, T, *tok_logvar.shape[1:])
        else:
            tok_mu_bt = None
            tok_logvar_bt = None

        if self._temporal_backend == "minimal_sru":
            # ---- SRU full-sequence path ----
            # Build SRU input tokens: (B, T, V + A + 1) = (B, T, 36)
            x_sru = torch.cat([h_spatial_bt, prev_actions, keep_float], dim=-1)

            z_all, z_last = self.world_hd.forward_sequence(
                x_sru, initial_state=temporal_state,
                valid_step=valid_step, return_all=True,
            )

            # ----- Vectorised controller reward over B*T positions -----
            out_bt = z_all.reshape(B * T, -1)            # (B*T, D)
            curr_act_bt = current_actions.reshape(B * T, -1)  # (B*T, A)
            _shared_bt, r_pred_bt = self.controller(out_bt, curr_act_bt)
            reward_pred_seq = r_pred_bt.view(B, T)          # (B, T)

            rp_last = reward_pred_seq[:, -1:].contiguous()

            # Placeholder history for backward compat.
            compat_history = torch.cat([h_spatial_bt, prev_actions], dim=-1)

            return WorldModelOutput(
                world_state=z_last,
                reward_pred=rp_last,
                mask_soft=selection_mask_bt[:, -1],
                indices=indices_bt[:, -1],
                history=compat_history,
                lengths=torch.full((B,), T, device=device, dtype=torch.long),
                tok_mu=tok_mu_bt,
                tok_logvar=tok_logvar_bt,
                reward_pred_seq=reward_pred_seq,
                temporal_state=z_last,
            )

        # ---- Causal Transformer full-sequence path (unchanged) ----
        # ----- Build Transformer tokens -----
        all_tokens = torch.cat([h_spatial_bt, prev_actions], dim=-1)  # (B, T, input_dim)

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
            temporal_state=None,
        )

    # ------------------------------------------------------------------
    # SRU blind imagination helpers
    # ------------------------------------------------------------------

    def blind_sru_step(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """One blind SRU step: build x_t with zero spatial, keep_bit=0, then step.

        Parameters
        ----------
        z_t:
            ``(B, D)`` — current recurrent state.
        action:
            ``(B, A)`` — action to append.

        Returns
        -------
        z_{t+1}:
            ``(B, D)`` — next recurrent state.
        """
        B = z_t.shape[0]
        device = z_t.device
        zero_spatial = torch.zeros(B, VALUES_DIM, device=device, dtype=z_t.dtype)
        keep_bit = torch.zeros(B, 1, device=device, dtype=z_t.dtype)
        x_t = torch.cat([zero_spatial, action, keep_bit], dim=-1)  # (B, 36)
        return self.world_hd.step(x_t, z_prev=z_t)

    def get_sru_warmup_beliefs(
        self,
        obs: torch.Tensor,
        prev_actions: torch.Tensor,
        current_actions: torch.Tensor,
        force_keep_input: bool = True,
        valid_step: Optional[torch.Tensor] = None,
        observation_keep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-step SRU states ``(B, T, D)`` for a warmup sequence.

        Mirrors the SRU path of ``forward_sequence`` but returns all
        intermediate ``z_t`` values, not just the final state.
        """
        B, T = obs.shape[0], obs.shape[1]
        device = obs.device

        frames = obs.reshape(B * T, *obs.shape[2:])
        feat = self.encoder(frames)
        tok_out = self.tokenizer(feat)
        tokens = tok_out if isinstance(tok_out, torch.Tensor) else tok_out[0]
        logits = self.scorer(tokens)
        selection_mask, indices = self.selector(logits)
        h_spatial, _attn = self.spatial_hd(tokens, logits, selection_mask, indices)
        h_spatial = self.obs_drop(h_spatial, force_keep=force_keep_input)

        if observation_keep is None:
            keep_float = torch.ones(
                B, T, 1, device=device, dtype=h_spatial.dtype,
            )
        else:
            if observation_keep.dtype != torch.bool:
                raise TypeError("observation_keep must have dtype bool")
            if observation_keep.shape != (B, T):
                raise ValueError(
                    "observation_keep must have shape "
                    f"{(B, T)}, got {tuple(observation_keep.shape)}"
                )
            keep_float = observation_keep.to(
                device=device, dtype=h_spatial.dtype,
            ).unsqueeze(-1)

        h_spatial_bt = h_spatial.view(B, T, -1) * keep_float

        x_sru = torch.cat([h_spatial_bt, prev_actions, keep_float], dim=-1)
        z_all, _ = self.world_hd.forward_sequence(x_sru, valid_step=valid_step, return_all=True)
        return z_all  # (B, T, D)

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
