"""Shared controller trunk with reward head.

Timing contract (approved):

    belief b_t = Transformer(obs[t], action[t-1], history)
    shared_repr = ControllerTrunk.encode(b_t)
    Actor(shared_repr)            → action[t]          (Stage 4)
    Critic(shared_repr)           → V(b_t)             (Stage 4)
    ControllerTrunk.predict_reward(shared_repr, action[t])
                                  → reward[t] (= r_{t+1})

The shared representation is ``actor-ready``: it depends only on the past
(observations up to ``t`` and actions up to ``t-1``), so the Actor can
choose ``action[t]`` without circular dependence.  The reward head
receives the current action separately.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from rwm.config.config import ACTION_DIM, WORLD_STATE_DIM


class ControllerTrunk(nn.Module):
    """Shared interpretation layer for the causal temporal state.

    Parameters
    ----------
    input_dim:
        Dimension of the incoming world state (default: ``WORLD_STATE_DIM``).
    hidden_dim:
        Hidden dimension of the shared trunk (default: ``input_dim``).
    action_dim:
        Dimension of the action vector for reward conditioning.
    """

    def __init__(
        self,
        input_dim: int = WORLD_STATE_DIM,
        hidden_dim: Optional[int] = None,
        action_dim: int = ACTION_DIM,
    ):
        super().__init__() # type: ignore
        if hidden_dim is None:
            hidden_dim = input_dim

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        # Reward head conditioned on shared_repr + current_action.
        self.reward_head = nn.Linear(hidden_dim + action_dim, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, belief: torch.Tensor) -> torch.Tensor:
        """Compute the actor-ready shared representation from a belief.

        Parameters
        ----------
        belief:
            ``(B, D)`` — latent temporal state from the CausalTransformer
            (depends only on the past).

        Returns
        -------
        shared_repr:
            ``(B, hidden_dim)`` — shared representation that the Actor and
            Critic will consume (Stage 4).  Pre-action: depends only on
            obs up to ``t`` and actions up to ``t-1``.
        """
        return self.shared(belief)

    def predict_reward(
        self,
        shared_repr: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict immediate reward from shared representation and the
        *current* action.

        Parameters
        ----------
        shared_repr:
            ``(B, hidden_dim)`` — from ``encode()``.
        action:
            ``(B, action_dim)`` — the action taken *after* the belief was
            computed (``action[t]``).

        Returns
        -------
        reward_pred:
            ``(B, 1)`` — predicted immediate reward ``r_{t+1}``.
        """
        x = torch.cat([shared_repr, action], dim=-1)
        return self.reward_head(x)

    def forward(
        self,
        belief: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience forward: encode + predict_reward in one call.

        Parameters
        ----------
        belief:
            ``(B, D)`` — latent temporal state.
        action:
            ``(B, action_dim)`` — current action.

        Returns
        -------
        shared_repr:
            ``(B, hidden_dim)`` — actor-ready shared representation.
        reward_pred:
            ``(B, 1)`` — predicted immediate reward.
        """
        h = self.encode(belief)
        r = self.predict_reward(h, action)
        return h, r
