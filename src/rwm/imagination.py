"""Differentiable bounded-context imagination interface (Stage 3.0 / S5.0).

Warm up from observed frames, then score-and-advance through imagined steps
using either the causal Transformer or MinimalSRU.

Causal semantics (Stage 3.0)::

    z_t = Transformer(H_t)              — belief from causal token history
    r_hat[t] = RewardHead(z_t, a_t)     — score before advancing
    H_(t+1) = append(cat(zeros, a_t))   — blind advance (no image)

SRU semantics (S5.0)::

    z_t = MinimalSRU.step(x_t, z_{t-1})  — recurrent state
    r_hat[t] = RewardHead(z_t, a_t)     — score before advancing
    z_{t+1} = blind_sru_step(z_t, a_t)  — blind advance (z only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from rwm.config.config import ACTION_DIM, VALUES_DIM, SEQ_LEN
from rwm.models.rwm.model import ReducedWorldModel
from rwm.utils.history_buffer import HistoryBuffer


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ImaginationState:
    """Causal or SRU state after observed warmup.

    Fields
    ------
    history:
        ``(B, T, V+A)`` — all tokens from the warmup sequence (causal) or
        placeholder ``(B, 1, V+A)`` (SRU, not meaningful).
    lengths:
        ``(B,)`` — ``T`` for causal, ``1`` for SRU (placeholder).
    beliefs:
        ``(B, T, D)`` — per-step beliefs (causal Transformer output) or
        per-step SRU states ``z_t``.
    is_sru:
        ``True`` if this state was produced by an SRU backend.
    """
    history: Tensor
    lengths: Tensor
    beliefs: Tensor
    is_sru: bool = False

    @property
    def current_belief(self) -> Tensor:
        """``(B, D)`` — belief at the last warmup step."""
        return self.beliefs[:, -1, :]


@dataclass
class RolloutOutput:
    """Output of a differentiable imagined rollout.

    Fields
    ------
    states:
        ``(B, H, D)`` — per-step beliefs *before* advancing
        (``z_t`` in the score-then-advance contract).
    actions:
        ``(B, H, A)`` — actions used at each imagined step.
    rewards:
        ``(B, H)`` — predicted rewards ``r_hat[t] = Reward(z_t, a_t)``.
    next_state:
        ``(B, D)`` — belief after the final imagined step.
    history:
        ``(B, T_total, V+A)`` — accumulated history after all steps
        (causal) or placeholder ``(B, 1, V+A)`` (SRU).
    lengths:
        ``(B,)`` — valid lengths after all steps.
    is_sru:
        ``True`` for SRU backends.
    """
    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_state: Tensor
    history: Tensor
    lengths: Tensor
    is_sru: bool = False


# ---------------------------------------------------------------------------
# Imagination interface
# ---------------------------------------------------------------------------

class ImaginationRollout(nn.Module):
    """Differentiable bounded-context imagination interface.

    Parameters
    ----------
    model:
        A ``ReducedWorldModel`` instance.
    """

    def __init__(self, model: ReducedWorldModel) -> None:
        super().__init__()
        self.model = model
        self._is_sru = model._temporal_backend == "minimal_sru"

    # ------------------------------------------------------------------
    # Warmup from observed frames
    # ------------------------------------------------------------------

    def warmup(
        self,
        obs: Tensor,                # (B, T, C, H, W)
        prev_actions: Tensor,       # (B, T, A)
        current_actions: Optional[Tensor] = None,  # (B, T, A) or None
        force_keep_input: bool = True,
        observation_keep: Optional[Tensor] = None,  # (B, T) bool
        valid_step: Optional[Tensor] = None,  # (B, T) bool — SRU only
    ) -> ImaginationState:
        """Run observed warmup and return the imagination state."""
        if current_actions is None:
            current_actions = torch.zeros(
                obs.shape[0], obs.shape[1], ACTION_DIM,
                device=obs.device, dtype=obs.dtype,
            )

        if self._is_sru:
            return self._warmup_sru(obs, prev_actions, current_actions,
                                    force_keep_input, observation_keep, valid_step)

        # Causal warmup (unchanged).
        out = self.model.forward_sequence(
            obs, prev_actions, current_actions,
            force_keep_input=force_keep_input,
            observation_keep=observation_keep,
        )
        beliefs = self.model.world_hd(
            out.history, lengths=out.lengths, return_all=True,
        )
        return ImaginationState(
            history=out.history, lengths=out.lengths, beliefs=beliefs, is_sru=False,
        )

    def _warmup_sru(
        self,
        obs: Tensor, prev_actions: Tensor, current_actions: Tensor,
        force_keep_input: bool, observation_keep: Optional[Tensor],
        valid_step: Optional[Tensor] = None,
    ) -> ImaginationState:
        """SRU warmup: extract per-step z_t states."""
        beliefs = self.model.get_sru_warmup_beliefs(
            obs, prev_actions, current_actions,
            force_keep_input=force_keep_input,
            valid_step=valid_step,
            observation_keep=observation_keep,
        )
        # Placeholder history for compatibility.
        B = obs.shape[0]
        device = obs.device
        placeholder_hist = torch.zeros(B, 1, VALUES_DIM + ACTION_DIM, device=device)
        placeholder_lens = torch.ones(B, dtype=torch.long, device=device)
        return ImaginationState(
            history=placeholder_hist, lengths=placeholder_lens,
            beliefs=beliefs, is_sru=True,
        )

    # ------------------------------------------------------------------
    # Score — predict reward without advancing
    # ------------------------------------------------------------------

    def score(self, belief: Tensor, action: Tensor) -> Tensor:
        """Predict reward ``r_hat[t+1] = Reward(z_t, a_t)``."""
        h = self.model.controller.encode(belief)
        return self.model.controller.predict_reward(h, action)

    # ------------------------------------------------------------------
    # Advance — one blind step
    # ------------------------------------------------------------------

    def advance(
        self,
        history: Tensor,   # (B, T_prev, V+A) — causal only
        lengths: Tensor,   # (B,) — causal only
        action: Tensor,    # (B, A)
        temporal_state: Optional[Tensor] = None,  # (B, D) — SRU only
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Advance one blind (no-image) step.

        SRU path (when ``temporal_state is not None``)::

            z_{t+1} = blind_sru_step(z_t, action)

        Causal path::

            H_(t+1) = append(cat(zeros, action), H_t)
            z_{t+1} = Transformer(H_(t+1))

        Returns
        -------
        new_history, new_lengths, new_belief
        """
        if self._is_sru and temporal_state is None:
            raise ValueError(
                "SRU advance requires an explicit temporal_state. "
                "Pass the previous SRU state."
            )
        if not self._is_sru and temporal_state is not None:
            raise ValueError(
                "Causal advance does not accept temporal_state. "
                "Pass history and lengths instead."
            )
        if temporal_state is not None:
            # SRU blind advance — no history needed.
            new_z = self.model.blind_sru_step(temporal_state, action)
            B = action.shape[0]
            device = action.device
            placeholder_hist = torch.zeros(B, 1, VALUES_DIM + ACTION_DIM, device=device)
            placeholder_lens = torch.ones(B, dtype=torch.long, device=device)
            return placeholder_hist, placeholder_lens, new_z

        # Causal blind advance (unchanged).
        B = action.shape[0]
        device = action.device
        dtype = history.dtype
        input_dim = VALUES_DIM + ACTION_DIM

        zero_spatial = torch.zeros(B, VALUES_DIM, device=device, dtype=dtype)
        blind_token = torch.cat([zero_spatial, action], dim=-1).unsqueeze(1)

        buf = HistoryBuffer.from_history(SEQ_LEN, input_dim, history, lengths)
        new_history, new_lengths = buf.append(blind_token)
        new_belief = self.model.world_hd(new_history, lengths=new_lengths)
        return new_history, new_lengths, new_belief

    # ------------------------------------------------------------------
    # Multi-step differentiable rollout
    # ------------------------------------------------------------------

    def rollout(
        self,
        history: Tensor,       # (B, T_warm, V+A) — causal only
        lengths: Tensor,       # (B,) — causal only
        initial_belief: Tensor,  # (B, D)
        actions: Tensor,       # (B, H, A)
        temporal_state: Optional[Tensor] = None,  # (B, D) — SRU only
    ) -> RolloutOutput:
        """Differentiable multi-step imagined rollout.

        For each step ``h`` in ``H``:

            1. **Score**:  ``r_hat[h] = Reward(belief, actions[:, h])``
            2. **Record**: store the belief and reward.
            3. **Advance**: update state (causal: append token; SRU: blind step).

        Parameters
        ----------
        temporal_state:
            Initial SRU state.  When provided, the SRU path is used.
        """
        B, H = actions.shape[0], actions.shape[1]
        device = actions.device
        d_model = initial_belief.shape[-1]

        if self._is_sru and temporal_state is None:
            raise ValueError(
                "SRU rollout requires an explicit temporal_state. "
                "Pass the initial SRU state."
            )
        if not self._is_sru and temporal_state is not None:
            raise ValueError(
                "Causal rollout does not accept temporal_state. "
                "Pass history and lengths instead."
            )

        states = torch.empty(B, H, d_model, device=device, dtype=initial_belief.dtype)
        rewards = torch.empty(B, H, device=device, dtype=initial_belief.dtype)

        belief = initial_belief
        curr_hist = history
        curr_lens = lengths
        curr_z = temporal_state  # None for causal

        for h in range(H):
            a_t = actions[:, h, :]  # (B, A)

            r_t = self.score(belief, a_t)  # (B, 1)

            states[:, h, :] = belief
            rewards[:, h] = r_t.squeeze(-1)

            curr_hist, curr_lens, belief = self.advance(
                curr_hist, curr_lens, a_t, temporal_state=curr_z,
            )
            if curr_z is not None:
                curr_z = belief  # SRU: belief IS the next z

        is_sru = (temporal_state is not None)
        return RolloutOutput(
            states=states, actions=actions, rewards=rewards,
            next_state=belief, history=curr_hist, lengths=curr_lens,
            is_sru=is_sru,
        )
