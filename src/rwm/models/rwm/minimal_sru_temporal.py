"""MinimalSRUTemporal — compact single-gate recurrent cell.

Equations
---------
::

    x_t = concat(observation_keep_t * spatial_t, previous_action_t, observation_keep_t)   # 36
    p_t = W_p @ x_t + b_p                                    # Linear(36 → 160)
    candidate_t  = tanh(p_t[..., :80])                       # (B, 80)
    carry_t      = sigmoid(p_t[..., 80:] + carry_bias_init)  # (B, 80)
    z_candidate  = carry_t * z_{t-1} + (1 - carry_t) * candidate_t
    z_t          = where(valid_step_t, z_candidate, z_{t-1})  # padding guard

Key semantics
-------------
- ``observation_keep=False`` means spatial input is zeroed, but actions and the
  visibility bit remain available → z MUST update.
- ``valid_step=False`` means padding → z MUST remain unchanged.
- These two masks are separate and must never be reused.
- ``z_t`` is both the complete recurrent state and the input to
  ``ControllerTrunk.encode()``.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class MinimalSRUTemporal(nn.Module):
    """Minimal single-gate recurrent cell.

    Parameters
    ----------
    input_dim:
        Dimension of the input token ``x_t`` (default 36).
    state_dim:
        Dimension of the recurrent state ``z_t`` (default 80).
    carry_bias_init:
        Initial bias for the carry gate sigmoid (default 1.0).
    """

    def __init__(
        self,
        input_dim: int = 36,
        state_dim: int = 80,
        carry_bias_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self._carry_bias_init = carry_bias_init

        self.projection = nn.Linear(input_dim, state_dim * 2)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def carry_bias_init(self) -> float:
        return self._carry_bias_init

    # ------------------------------------------------------------------
    # Input validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_x_t(x_t: Tensor, input_dim: int) -> None:
        if x_t.dim() != 2 or x_t.size(-1) != input_dim:
            raise ValueError(
                f"x_t must be (B, {input_dim}), got {tuple(x_t.shape)}"
            )

    @staticmethod
    def _validate_z_prev(z_prev: Tensor, state_dim: int, label: str = "z_prev") -> None:
        if z_prev.dim() != 2 or z_prev.size(-1) != state_dim:
            raise ValueError(
                f"{label} must be (B, {state_dim}), got {tuple(z_prev.shape)}"
            )

    @staticmethod
    def _validate_valid_step(valid_step: Tensor, B: int, label: str = "valid_step") -> None:
        if valid_step.dtype != torch.bool:
            raise ValueError(
                f"{label} must be bool, got {valid_step.dtype}"
            )
        if valid_step.dim() != 1 or valid_step.size(0) != B:
            raise ValueError(
                f"{label} must be ({B},), got {tuple(valid_step.shape)}"
            )

    @staticmethod
    def _validate_x_seq(x: Tensor, input_dim: int) -> None:
        if x.dim() != 3 or x.size(-1) != input_dim:
            raise ValueError(
                f"x must be (B, T, {input_dim}), got {tuple(x.shape)}"
            )
        if x.size(1) < 1:
            raise ValueError(
                f"x must have T >= 1, got T={x.size(1)}"
            )

    @staticmethod
    def _validate_valid_step_seq(valid_step: Tensor, B: int, T: int) -> None:
        if valid_step.dtype != torch.bool:
            raise ValueError(f"valid_step must be bool, got {valid_step.dtype}")
        if valid_step.shape != (B, T):
            raise ValueError(
                f"valid_step must be ({B}, {T}), got {tuple(valid_step.shape)}"
            )

    # ------------------------------------------------------------------
    # Single-step inference
    # ------------------------------------------------------------------

    def step(
        self,
        x_t: Tensor,                 # (B, input_dim)
        z_prev: Optional[Tensor] = None,  # (B, state_dim); zeros if None
        valid_step: Optional[Tensor] = None,  # (B,) bool; True=normal, False=padding
    ) -> Tensor:
        """Single recurrent step.

        Parameters
        ----------
        x_t:
            Input token for this step.
        z_prev:
            Previous recurrent state.  If ``None``, treated as zeros
            (episode start).
        valid_step:
            ``True`` = normal update, ``False`` = padding (state unchanged).
            If ``None``, all steps are treated as valid.

        Returns
        -------
        z_t:
            ``(B, state_dim)`` — updated recurrent state.
        """
        self._validate_x_t(x_t, self.input_dim)
        B = x_t.shape[0]

        if z_prev is not None:
            self._validate_z_prev(z_prev, self.state_dim, "z_prev")

        if valid_step is not None:
            self._validate_valid_step(valid_step, B, "valid_step")

        device = x_t.device

        if z_prev is None:
            z_prev = torch.zeros(B, self.state_dim, device=device, dtype=x_t.dtype)

        p = self.projection(x_t)  # (B, 2 * state_dim)
        candidate = torch.tanh(p[..., :self.state_dim])
        carry = torch.sigmoid(p[..., self.state_dim:] + self._carry_bias_init)

        z_candidate = carry * z_prev + (1.0 - carry) * candidate

        if valid_step is not None:
            vs = valid_step.unsqueeze(-1).to(dtype=x_t.dtype)
            z_t = torch.where(vs.bool(), z_candidate, z_prev)
        else:
            z_t = z_candidate

        return z_t

    # ------------------------------------------------------------------
    # Full-sequence (parallel projection, sequential recurrence)
    # ------------------------------------------------------------------

    def forward_sequence(
        self,
        x: Tensor,                       # (B, T, input_dim)
        initial_state: Optional[Tensor] = None,  # (B, state_dim); zeros if None
        valid_step: Optional[Tensor] = None,     # (B, T) bool
        return_all: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Full-sequence forward pass.

        The linear projection is computed once over flattened ``(B*T, input_dim)``,
        then the elementwise recurrence is scanned sequentially over ``T``.
        The projection accounts for the dominant parameters; the scan loop
        performs only elementwise operations and is NOT a parallel scan — no
        claim is made of original CUDA SRU parallel speedup.

        Parameters
        ----------
        x:
            Input tokens ``(B, T, input_dim)``.
        initial_state:
            Initial recurrent state.  ``None`` → zeros.
        valid_step:
            ``(B, T)`` boolean mask.  ``None`` → all valid.
        return_all:
            If ``True``, return ``(states, final_state)``.
            If ``False``, return only ``(empty, final_state)`` where
            ``states`` is ``None`` (caller uses ``final_state``).

        Returns
        -------
        states:
            ``(B, T, state_dim)`` or ``None`` (see ``return_all``).
        final_state:
            ``(B, state_dim)`` — state after the last valid step.
        """
        self._validate_x_seq(x, self.input_dim)
        B, T = x.shape[0], x.shape[1]

        if initial_state is not None:
            self._validate_z_prev(initial_state, self.state_dim, "initial_state")

        if valid_step is not None:
            self._validate_valid_step_seq(valid_step, B, T)

        device = x.device

        # Fused projection over flattened batch-time.
        flat_x = x.reshape(B * T, -1)                 # (B*T, input_dim)
        p = self.projection(flat_x)                    # (B*T, 2 * state_dim)
        p = p.view(B, T, -1)                           # (B, T, 2 * state_dim)

        candidate = torch.tanh(p[..., :self.state_dim])   # (B, T, state_dim)
        carry = torch.sigmoid(p[..., self.state_dim:] + self._carry_bias_init)

        z = initial_state
        if z is None:
            z = torch.zeros(B, self.state_dim, device=device, dtype=x.dtype)

        if return_all:
            states_l = []
            for t in range(T):
                z_candidate = carry[:, t, :] * z + (1.0 - carry[:, t, :]) * candidate[:, t, :]
                if valid_step is not None:
                    vs = valid_step[:, t].unsqueeze(-1).to(dtype=x.dtype)
                    z = torch.where(vs.bool(), z_candidate, z)
                else:
                    z = z_candidate
                states_l.append(z)
            states = torch.stack(states_l, dim=1)
            return states, z
        else:
            for t in range(T):
                z_candidate = carry[:, t, :] * z + (1.0 - carry[:, t, :]) * candidate[:, t, :]
                if valid_step is not None:
                    vs = valid_step[:, t].unsqueeze(-1).to(dtype=x.dtype)
                    z = torch.where(vs.bool(), z_candidate, z)
                else:
                    z = z_candidate
            return None, z
