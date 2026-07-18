"""Differentiable bounded-context imagination interface (Stage 3.0).

Warm up from observed frames, then score-and-advance through imagined steps
using only the causal Transformer and ControllerTrunk.  No perception/CNN is
called on dummy images during imagined steps.

State semantics (from the transition contract):

    z_t = Transformer(H_t)              — belief from causal token history
    r_hat[t] = RewardHead(z_t, a_t)     — score before advancing
    H_(t+1) = append(cat(zeros, a_t))   — blind advance (no image)

Never uses ``torch.no_grad()``, ``.item()``, or ``.detach()`` in trainable
rollout paths, preserving end-to-end differentiability.
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
    """Causal state after observed warmup.

    Fields
    ------
    history:
        ``(B, T, V+A)`` — all tokens from the warmup sequence.
    lengths:
        ``(B,)`` — ``T`` for every batch element.
    beliefs:
        ``(B, T, D)`` — per-step causal beliefs (before controller).
    """
    history: Tensor
    lengths: Tensor
    beliefs: Tensor

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
        ``(B, T_total, V+A)`` — accumulated history after all steps.
    lengths:
        ``(B,)`` — valid lengths after all steps.
    """
    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_state: Tensor
    history: Tensor
    lengths: Tensor


# ---------------------------------------------------------------------------
# Imagination interface
# ---------------------------------------------------------------------------

class ImaginationRollout(nn.Module):
    """Differentiable bounded-context imagination interface.

    Owns a reference to the world model (not the model itself).  All
    tensor operations preserve gradient flow and never call ``item()``,
    ``detach()``, or ``torch.no_grad()``.

    Parameters
    ----------
    model:
        A ``ReducedWorldModel`` instance whose ``world_hd`` and
        ``controller`` are used for score-and-advance.
    """

    def __init__(self, model: ReducedWorldModel) -> None:
        super().__init__()
        self.model = model

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
    ) -> ImaginationState:
        """Run observed warmup and return the causal state.

        Uses ``forward_sequence`` for vectorised perception and a
        re-entrant transformer call to extract per-step beliefs.
        When ``current_actions`` is ``None``, zeros are used only to satisfy
        the existing full-sequence API; warmup reward predictions are ignored.
        """
        if current_actions is None:
            current_actions = torch.zeros(
                obs.shape[0], obs.shape[1], ACTION_DIM,
                device=obs.device, dtype=obs.dtype,
            )

        out = self.model.forward_sequence(
            obs, prev_actions, current_actions,
            force_keep_input=force_keep_input,
            observation_keep=observation_keep,
        )

        beliefs = self.model.world_hd(
            out.history, lengths=out.lengths, return_all=True,
        )

        return ImaginationState(
            history=out.history,
            lengths=out.lengths,
            beliefs=beliefs,
        )

    # ------------------------------------------------------------------
    # Score — predict reward without advancing
    # ------------------------------------------------------------------

    def score(self, belief: Tensor, action: Tensor) -> Tensor:
        """Predict reward ``r_hat[t+1] = Reward(z_t, a_t)``.

        Parameters
        ----------
        belief:
            ``(B, D)`` — causal belief ``z_t``.
        action:
            ``(B, A)`` — candidate action ``a_t``.

        Returns
        -------
        reward:
            ``(B, 1)`` — predicted reward for taking ``a_t`` at belief ``z_t``.
        """
        h = self.model.controller.encode(belief)
        return self.model.controller.predict_reward(h, action)

    # ------------------------------------------------------------------
    # Advance — one blind step
    # ------------------------------------------------------------------

    def advance(
        self,
        history: Tensor,   # (B, T_prev, V+A)
        lengths: Tensor,   # (B,)
        action: Tensor,    # (B, A)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Advance one blind (no-image) step.

        Builds token ``cat(zero_spatial, action)``, appends it to the
        history buffer (with left truncation), runs the causal Transformer,
        and returns the updated history, lengths, and new belief.

        Parameters
        ----------
        history:
            Current token history ``(B, T_prev, V+A)``.
        lengths:
            ``(B,)`` — valid lengths in ``history``.
        action:
            ``(B, A)`` — action to append as the *previous* action for the
            next state.

        Returns
        -------
        new_history:
            ``(B, T_new, V+A)`` — ``T_new = min(T_prev + 1, SEQ_LEN)``.
        new_lengths:
            ``(B,)`` — ``T_new`` for every batch element.
        new_belief:
            ``(B, D)`` — causal belief at the new last position.
        """
        B = action.shape[0]
        device = action.device
        dtype = history.dtype
        input_dim = VALUES_DIM + ACTION_DIM

        zero_spatial = torch.zeros(B, VALUES_DIM, device=device, dtype=dtype)
        blind_token = torch.cat([zero_spatial, action], dim=-1).unsqueeze(1)

        buf = HistoryBuffer.from_history(
            SEQ_LEN, input_dim, history, lengths,
        )
        new_history, new_lengths = buf.append(blind_token)

        new_belief = self.model.world_hd(new_history, lengths=new_lengths)
        return new_history, new_lengths, new_belief

    # ------------------------------------------------------------------
    # Multi-step differentiable rollout
    # ------------------------------------------------------------------

    def rollout(
        self,
        history: Tensor,       # (B, T_warm, V+A)
        lengths: Tensor,       # (B,)
        initial_belief: Tensor,  # (B, D)
        actions: Tensor,       # (B, H, A)
    ) -> RolloutOutput:
        """Differentiable multi-step imagined rollout.

        For each step ``h`` in ``H``:

            1. **Score**:  ``r_hat[h] = Reward(belief, actions[:, h])``
            2. **Record**: store the belief and reward.
            3. **Advance**: append ``cat(zeros, actions[:, h])``, update
               history, and compute the next belief.

        Parameters
        ----------
        history:
            Token history from warmup ``(B, T_warm, V+A)``.
        lengths:
            ``(B,)`` — valid lengths in ``history``.
        initial_belief:
            ``(B, D)`` — belief at the last warmup step (``z_0``).
        actions:
            ``(B, H, A)`` — imagined action sequence.

        Returns
        -------
        ``RolloutOutput`` with fields described in the class docstring.
        """
        B, H = actions.shape[0], actions.shape[1]
        device = actions.device
        d_model = initial_belief.shape[-1]

        states = torch.empty(B, H, d_model, device=device, dtype=initial_belief.dtype)
        rewards = torch.empty(B, H, device=device, dtype=initial_belief.dtype)

        belief = initial_belief
        curr_hist = history
        curr_lens = lengths

        for h in range(H):
            a_t = actions[:, h, :]  # (B, A)

            r_t = self.score(belief, a_t)  # (B, 1)

            states[:, h, :] = belief
            rewards[:, h] = r_t.squeeze(-1)

            curr_hist, curr_lens, belief = self.advance(curr_hist, curr_lens, a_t)

        return RolloutOutput(
            states=states,
            actions=actions,
            rewards=rewards,
            next_state=belief,
            history=curr_hist,
            lengths=curr_lens,
        )
