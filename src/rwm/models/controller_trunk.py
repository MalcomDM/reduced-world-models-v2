"""Shared controller trunk with configurable reward head.

Timing contract (approved):

    belief b_t = Transformer(obs[t], action[t-1], history)
    shared_repr = ControllerTrunk.encode(b_t)
    Actor(shared_repr)            → action[t]          (Stage 4)
    Critic(shared_repr)           → V(b_t)             (Stage 4)
    ControllerTrunk.predict_reward(shared_repr, action[t])
                                  → reward[t] (= r_{t+1})

Reward head architecture is configurable:

    "linear" (default):
        reward = Linear(concat(c_t, action_t)) -> scalar

    "nonlinear":
        reward = Linear(concat(c_t, action_t), hidden)
                 -> ReLU
                 -> Linear(hidden, 1)

The shared representation ``c_t = encode(belief)`` is actor-ready: it depends
only on the past, so the Actor can choose action[t] without circular
dependence.  The reward head receives the current action separately.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from rwm.config.config import ACTION_DIM, WORLD_STATE_DIM

_REWARD_HEAD_KINDS = ("linear", "nonlinear")


class ControllerTrunk(nn.Module):
    """Shared interpretation layer for the causal temporal state.

    Parameters
    ----------
    input_dim:
        Dimension of the incoming world state.
    hidden_dim:
        Hidden dimension of the shared trunk (default: ``input_dim``).
    action_dim:
        Dimension of the action vector for reward conditioning.
    reward_head_kind:
        ``"linear"`` (default) or ``"nonlinear"``.
    reward_head_hidden_dim:
        Hidden dimension for the nonlinear head (ignored for linear).
        Default: 32.
    """

    def __init__(
        self,
        input_dim: int = WORLD_STATE_DIM,
        hidden_dim: Optional[int] = None,
        action_dim: int = ACTION_DIM,
        reward_head_kind: str = "linear",
        reward_head_hidden_dim: int = 32,
    ):
        super().__init__()

        if reward_head_kind not in _REWARD_HEAD_KINDS:
            raise ValueError(
                f"reward_head_kind must be one of {_REWARD_HEAD_KINDS}, "
                f"got {reward_head_kind!r}"
            )
        if not isinstance(reward_head_hidden_dim, int) or reward_head_hidden_dim < 1:
            raise ValueError(
                "reward_head_hidden_dim must be a positive integer, "
                f"got {reward_head_hidden_dim!r}"
            )
        self._reward_head_kind = reward_head_kind

        if hidden_dim is None:
            hidden_dim = input_dim

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        joint_dim = hidden_dim + action_dim
        if reward_head_kind == "linear":
            self.reward_head = nn.Linear(joint_dim, 1)
        else:
            self.reward_head = nn.Sequential(
                nn.Linear(joint_dim, reward_head_hidden_dim),
                nn.ReLU(),
                nn.Linear(reward_head_hidden_dim, 1),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, belief: torch.Tensor) -> torch.Tensor:
        """Actor-ready shared representation from a belief."""
        return self.shared(belief)

    def predict_reward(
        self,
        shared_repr: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict immediate reward from shared representation and the
        *current* action."""
        x = torch.cat([shared_repr, action], dim=-1)
        return self.reward_head(x)

    def forward(
        self,
        belief: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience: encode + predict_reward in one call."""
        h = self.encode(belief)
        r = self.predict_reward(h, action)
        return h, r
