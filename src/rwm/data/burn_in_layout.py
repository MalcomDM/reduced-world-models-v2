"""Burn-in layout helper for SRU mid-window state initialisation.

Approved decision (S0): 20-step loss-masked same-episode burn-in.  This is a
*truncated* 20-step initial state, comparable with the causal Transformer
context window — NOT the exact full-episode state.

This module defines the layout contract only.  It does NOT modify the
Dataset, trainer, or loss functions — that integration is a separate step.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BurnInLayout:
    """Describes a burn-in + target window layout within one episode.

    The episode is treated as a 1-D array of length ``episode_len``.
    A training window of ``target_len`` steps starting at ``offset`` is
    preceded by a burn-in prefix of up to ``burn_in`` steps from the same
    episode.

    Fields
    ------
    offset:
        Window start position in the episode.
    burn_in:
        Number of burn-in steps requested (config parameter, typically 20).
    target_len:
        Number of target training steps (config parameter, typically 16).
    episode_len:
        Total length of the episode.
    burn_in_start:
        Integer index where the burn-in prefix begins (inclusive).
        May be 0 (episode start) if ``offset < burn_in``.
    burn_in_end:
        Integer index where the burn-in prefix ends (exclusive).
        Equal to ``offset``.
    target_start:
        Integer index where the target window begins (inclusive).
        Equal to ``offset``.
    target_end:
        Integer index where the target window ends (exclusive).
    total_start:
        Integer index of the first frame loaded (burn-in prefix start).
    total_end:
        Integer index of the last frame loaded (target window end).
    total_len:
        Total number of steps loaded = ``total_end - total_start``.
    effective_burn_in:
        Actual number of burn-in steps available (fewer at episode start).
    loss_mask_start:
        Position within the loaded ``[total_start, total_end)`` segment
        where the loss mask switches from ``False`` (burn-in, masked) to
        ``True`` (target, active).  This is ``offset - total_start``.
    """

    offset: int
    burn_in: int
    target_len: int
    episode_len: int

    burn_in_start: int
    burn_in_end: int
    target_start: int
    target_end: int
    total_start: int
    total_end: int
    total_len: int
    effective_burn_in: int
    loss_mask_start: int


def compute_burn_in_layout(
    offset: int,
    episode_len: int,
    burn_in: int = 20,
    target_len: int = 16,
) -> BurnInLayout:
    """Compute the burn-in layout for a given window offset.

    Parameters
    ----------
    offset:
        Window start position in the episode (0-indexed).
    episode_len:
        Total length of the episode.
    burn_in:
        Desired number of burn-in steps (default 20).
    target_len:
        Desired number of target training steps (default 16).

    Returns
    -------
    ``BurnInLayout`` with all derived positions.

    Examples
    --------
    Episode of length 100, burn_in=20, target_len=16:

    - offset 0: burn-in=[], target=[0, 16), loss mask starts at 0.
    - offset 5: burn-in=[0, 5), target=[5, 21), loss mask starts at 5.
    - offset 20: burn-in=[0, 20), target=[20, 36), loss mask starts at 20.
    - offset 50: burn-in=[30, 50), target=[50, 66), loss mask starts at 20.
    - offset 90 (near end): burn-in=[70, 90), target=[90, 100) truncated,
      loss mask starts at 20.
    """
    target_start = offset
    target_end = min(offset + target_len, episode_len)

    burn_in_start = max(0, offset - burn_in)
    burn_in_end = offset

    total_start = burn_in_start
    total_end = target_end

    effective_burn_in = burn_in_end - burn_in_start
    loss_mask_start = offset - total_start

    return BurnInLayout(
        offset=offset,
        burn_in=burn_in,
        target_len=target_len,
        episode_len=episode_len,
        burn_in_start=burn_in_start,
        burn_in_end=burn_in_end,
        target_start=target_start,
        target_end=target_end,
        total_start=total_start,
        total_end=total_end,
        total_len=total_end - total_start,
        effective_burn_in=effective_burn_in,
        loss_mask_start=loss_mask_start,
    )


def build_valid_step_mask(
    layout: BurnInLayout,
) -> list[bool]:
    """Build a per-step valid-step mask for the loaded segment.

    All steps in the burn-in + target range are valid (the data exists).
    This mask is ``True`` for every position because the current dataset
    does not contain padding.

    Returns a list of ``bool`` of length ``layout.total_len``, all ``True``.
    """
    return [True] * layout.total_len


def build_loss_mask(
    layout: BurnInLayout,
) -> list[bool]:
    """Build a per-step loss mask for the loaded segment.

    Positions in the burn-in prefix have loss masked (``False``).
    Positions in the target window have loss active (``True``).

    Returns a list of ``bool`` of length ``layout.total_len``.
    """
    mask = [False] * layout.effective_burn_in
    mask += [True] * (layout.total_len - layout.effective_burn_in)
    return mask


def build_source_position_map(
    layout: BurnInLayout,
) -> list[int]:
    """Build a list of source episode positions for each loaded step.

    Returns a list of ``int`` of length ``layout.total_len``, where each
    element is the episode index ``[total_start, total_end)``.
    """
    return list(range(layout.total_start, layout.total_end))
