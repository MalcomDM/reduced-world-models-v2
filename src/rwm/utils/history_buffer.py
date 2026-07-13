"""Shared temporal-history helper.

Owns append, truncation to context length, device/dtype handling, and
valid-length tracking.  Used by ``ReducedWorldModel``, evaluation,
rollout warmup, and imagined-rollout code.

Does **not** use ``torch.no_grad()`` — preserving gradient flow through
history is required for end-to-end behaviour gradients in Stage 2.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


class HistoryBuffer:
    """Append-only temporal history with bounded context length.

    Each element appended via ``append()`` is a single token of shape
    ``(B, 1, input_dim)``.  The buffer maintains a tensor ``(B, T, input_dim)``
    where ``T <= max_seq_len``.  When ``T`` would exceed ``max_seq_len`` the
    oldest tokens are dropped from the left.

    ``valid_lengths`` is a ``(B,)`` long tensor tracking the number of
    valid tokens per batch element. Padding, when needed for batched external
    sequences, is represented by explicit history and length inputs.

    Parameters
    ----------
    max_seq_len:
        Maximum number of tokens retained in the buffer.
    input_dim:
        Feature dimension of each token.
    device:
        Torch device for the buffer tensor.
    dtype:
        Torch dtype for the buffer tensor.
    """

    def __init__(
        self,
        max_seq_len: int,
        input_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._max_seq_len = max_seq_len
        self._input_dim = input_dim
        self._device = device
        self._dtype = dtype
        self._batch_size: int = 0
        self._history: Optional[Tensor] = None
        self._valid_lengths: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def history(self) -> Optional[Tensor]:
        """ ``(B, T, input_dim)`` or ``None`` if no tokens appended. """
        return self._history

    @property
    def valid_lengths(self) -> Optional[Tensor]:
        """ ``(B,)`` long tensor, or ``None`` if no tokens appended. """
        return self._valid_lengths

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    @property
    def input_dim(self) -> int:
        return self._input_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the buffer (drops all history)."""
        self._history = None
        self._valid_lengths = None

    def append(self, token: Tensor) -> Tuple[Tensor, Tensor]:
        """Append a single token and return the updated (history, lengths).

        Parameters
        ----------
        token:
            ``(B, 1, input_dim)`` tensor.

        Returns
        -------
        history:
            ``(B, T, input_dim)`` where ``T = min(prev_T + 1, max_seq_len)``.
        valid_lengths:
            ``(B,)`` long tensor with valid lengths.
        """
        B = token.shape[0]
        if self._batch_size == 0:
            self._batch_size = B

        if self._history is None:
            self._history = token
        else:
            self._history = torch.cat([self._history, token], dim=1)

        if self._history.size(1) > self._max_seq_len:
            self._history = self._history[:, -self._max_seq_len:, :]

        T = self._history.size(1)
        self._valid_lengths = torch.full(
            (B,), T, device=self._device, dtype=torch.long,
        )

        return self._history, self._valid_lengths

    @classmethod
    def from_history(
        cls,
        max_seq_len: int,
        input_dim: int,
        history: Optional[Tensor],
        lengths: Optional[Tensor],
    ) -> "HistoryBuffer":
        """Create a ``HistoryBuffer`` pre-populated from existing tensors.

        This is useful when restoring buffer state from a model's previous
        forward pass.

        Parameters
        ----------
        max_seq_len, input_dim:
            Buffer configuration (must match the original).
        history:
            ``(B, T, input_dim)`` or ``None``.
        lengths:
            ``(B,)`` long tensor or ``None``.
        """
        buf = cls(max_seq_len, input_dim, device=torch.device("cpu"))
        if history is not None:
            buf._batch_size = history.shape[0]
            buf._history = history
            buf._valid_lengths = lengths
            buf._device = history.device
            buf._dtype = history.dtype
        return buf
