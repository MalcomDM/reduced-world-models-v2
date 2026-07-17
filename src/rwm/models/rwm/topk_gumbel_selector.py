"""Top-K patch selector with configurable selection mode.

Modes:
  "learned" (default):
      Gumbel-Softmax + STE.  Gradient flows to unselected logits.
  "fixed_uniform":
      K positions chosen via deterministic farthest-point sampling on the
      patch grid.  Identical for every frame.  Preserves learned pooling
      weights within the fixed candidate set.
  "fixed_random":
      K patch positions sampled once from ``selection_seed``, then fixed.
      Identical for every frame at train and eval time.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rwm.config.config import K as _DEFAULT_K, PATCHES_PER_SIDE

_SELECTION_MODES = ("learned", "fixed_uniform", "fixed_random")
_N_PATCHES = PATCHES_PER_SIDE * PATCHES_PER_SIDE  # 225


# ---------------------------------------------------------------------------
# Farthest-point sampling on the 2-D patch grid
# ---------------------------------------------------------------------------

def _farthest_point_sample(k: int) -> Tensor:
    """Deterministic farthest-point sampling on a ``15×15`` grid.

    Returns ``(K,)`` flattened row-major indices with spatially spread
    coverage.  Starts from the grid centre, then iteratively picks the
    unselected grid cell whose minimum squared Euclidean distance to
    already selected cells is largest.  Ties broken by lowest flattened
    index.
    """
    side = PATCHES_PER_SIDE                  # 15
    total = side * side                       # 225

    # All grid positions as (y, x) in [0, side).
    ys = torch.arange(side, dtype=torch.float32).view(-1, 1).expand(side, side)
    xs = torch.arange(side, dtype=torch.float32).view(1, -1).expand(side, side)
    coords = torch.stack([ys, xs], dim=-1)    # (side, side, 2)

    flat_coords = coords.reshape(total, 2)    # (225, 2)

    # Start from the centre cell.
    centre = side // 2
    start_idx = centre * side + centre

    selected = [start_idx]
    selected_set = {start_idx}

    # Squared distances to nearest selected point.
    # Pre-allocate.
    min_dist = (flat_coords - flat_coords[start_idx]).pow(2).sum(dim=1)  # (225,)

    for _ in range(1, k):
        # Find the unselected point with maximum min-dist.
        # Exclude already selected (distance effectively 0).
        # Set selected distances to -inf so they can never win.
        dists = min_dist.clone()
        dists[list(selected_set)] = -1.0
        best = dists.argmax().item()

        selected.append(best)
        selected_set.add(best)

        # Update min distances for remaining points.
        new_dist = (flat_coords - flat_coords[best]).pow(2).sum(dim=1)
        min_dist = torch.min(min_dist, new_dist)

    return torch.tensor(selected[:k], dtype=torch.long)


def _fixed_random_indices(k: int, seed: int) -> Tensor:
    """Return ``(K,)`` of randomly selected patch indices (deterministic from seed)."""
    rng = torch.Generator().manual_seed(seed)
    return torch.randperm(_N_PATCHES, generator=rng)[:k]


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class TopKGumbelSelector(nn.Module):
    """Differentiable Top-K selector with configurable selection mode.

    Parameters
    ----------
    k:
        Number of selected tokens (1..225).
    temp:
        Temperature for Gumbel-Softmax and the soft surrogate.
    selection_mode:
        ``"learned"``, ``"fixed_uniform"``, or ``"fixed_random"``.
    selection_seed:
        RNG seed for ``fixed_random`` mode.
    """

    def __init__(
        self,
        k: int = _DEFAULT_K,
        temp: float = 1.0,
        selection_mode: str = "learned",
        selection_seed: int = 0,
    ) -> None:
        super().__init__()
        if selection_mode not in _SELECTION_MODES:
            raise ValueError(
                f"selection_mode must be one of {_SELECTION_MODES}, got {selection_mode!r}"
            )
        if not isinstance(k, int) or k < 1 or k > _N_PATCHES:
            raise ValueError(
                f"k must be an integer in [1, {_N_PATCHES}], got {k}."
            )
        self.k = k
        self.temp = temp
        self.selection_mode = selection_mode
        self._selection_seed = selection_seed

        # Non-persistent: reconstructed from mode/k/seed, never serialised.
        # A ``_load_from_state_dict`` hook handles legacy checkpoints that
        # contain a ``_fixed_indices`` buffer from the broken implementation.
        self._fixed_indices: Tensor | None = None

        self._build_fixed_indices()

    def _build_fixed_indices(self) -> None:
        if self.selection_mode == "fixed_uniform":
            self._fixed_indices = _farthest_point_sample(self.k)
        elif self.selection_mode == "fixed_random":
            self._fixed_indices = _fixed_random_indices(self.k, self._selection_seed)
        else:
            self._fixed_indices = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Legacy compat: pop ``_fixed_indices`` if present (from broken
        register_buffer period) and reconstruct from mode/seed/k."""
        key = prefix + "_fixed_indices"
        state_dict.pop(key, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._build_fixed_indices()

    def _sample_gumbel(self, shape: Tuple[int, ...], device: torch.device, eps: float = 1e-20) -> Tensor:
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, logits: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (selection_mask, indices)."""
        B, N = logits.shape
        device = logits.device

        if self.selection_mode == "learned":
            return self._forward_learned(logits, B, N, device)
        else:
            return self._forward_fixed(B, N, device, logits.dtype)

    def _forward_learned(self, logits: Tensor, B: int, N: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        if self.training:
            noise = self._sample_gumbel((B, N), device=device)
            topk_logits = (logits + noise) / self.temp
        else:
            topk_logits = logits

        indices = topk_logits.topk(self.k, dim=1).indices
        hard_mask = torch.zeros(B, N, device=device, dtype=logits.dtype)
        hard_mask.scatter_(1, indices, 1.0)

        soft = F.softmax(logits / self.temp, dim=1)
        soft_k = self.k * soft
        mask = (hard_mask - soft_k).detach() + soft_k
        return mask, indices

    def _forward_fixed(
        self,
        B: int,
        N: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        assert self._fixed_indices is not None
        fi = self._fixed_indices.to(device=device, dtype=torch.long)
        indices = fi.unsqueeze(0).expand(B, -1).contiguous()
        mask = torch.zeros(B, N, device=device, dtype=dtype)
        mask.scatter_(1, indices, 1.0)
        return mask, indices
