"""Temporal observation-mask sampler for Stage 2.5D.1 training.

Builds ``observation_keep (B, T)`` tensors with a contiguous masked block
after a configurable warmup.  Supports a ramp schedule for the per-sample
masking probability.
"""

from typing import List, Optional

import torch


def _validate_config(warmup_steps: int, horizons: List[int], sequence_len: int) -> None:
    """Validate mask configuration against sequence length.

    Raises ValueError if any horizon exceeds the remaining capacity after
    warmup, or if warmup >= sequence_len.
    """
    if warmup_steps >= sequence_len:
        raise ValueError(
            f"warmup_steps={warmup_steps} must be < sequence_len={sequence_len}"
        )
    max_allowed = sequence_len - warmup_steps
    for h in horizons:
        if h > max_allowed:
            raise ValueError(
                f"horizon {h} exceeds remaining capacity after warmup "
                f"(max {max_allowed}) for sequence_len={sequence_len}, "
                f"warmup_steps={warmup_steps}"
            )


def current_mask_probability(
    epoch: int,
    target_probability: float,
    ramp_epochs: int,
) -> float:
    """Linear ramp from 0 to target_probability over ramp_epochs.

    After ramp_epochs, returns target_probability.
    Before epoch 1 (epoch=0), returns 0.
    """
    if ramp_epochs <= 0 or epoch >= ramp_epochs:
        return target_probability
    return target_probability * (epoch / ramp_epochs)


def sample_mask(
    batch_size: int,
    sequence_len: int,
    warmup_steps: int,
    horizons: List[int],
    mask_probability: float,
    rng: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Sample ``observation_keep (B, T)`` bool tensor.

    For each sample in the batch, with probability ``mask_probability``,
    mask a contiguous block of length ``horizon`` (sampled uniformly from
    ``horizons``) starting at ``warmup_steps``.  Steps before warmup are
    always visible.  Steps after the masked block are visible again.

    With probability ``1 - mask_probability``, the sample is all-visible.
    """
    _validate_config(warmup_steps, horizons, sequence_len)
    keep = torch.ones(batch_size, sequence_len, dtype=torch.bool, device=device)

    if mask_probability <= 0:
        return keep

    # Generate on CPU (rng is always CPU), move to target device
    mask_rand = torch.rand(batch_size, 1, generator=rng)
    should_mask = mask_rand.to(device) < mask_probability  # (B, 1)

    if not should_mask.any():
        return keep

    # Per-sample horizon
    h_rand = torch.randint(len(horizons), (batch_size,), generator=rng)
    chosen_h = torch.tensor(horizons, device=device)[h_rand.to(device)]  # (B,)

    # Apply mask: set keep[:, warmup:warmup+horizon] = False for masked samples
    for b in range(batch_size):
        if should_mask[b].item():
            h = int(chosen_h[b].item())
            keep[b, warmup_steps:warmup_steps + h] = False

    return keep
