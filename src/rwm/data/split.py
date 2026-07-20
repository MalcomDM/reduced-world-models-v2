"""Reusable train/validation file-split helper.

Used by both the Stage 6.1 joint trainer and the checkpoint evaluator to
guarantee disjoint train/validation file sets.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from rwm.data.rollout_dataset import _collect_npz_files


def collect_and_split(
    root: Path,
    data_split_seed: int,
    val_ratio: float = 0.2,
) -> Tuple[List[Path], List[Path]]:
    """Deterministic file-level train/validation split.

    Parameters
    ----------
    root:
        Directory containing ``.npz`` rollout files (searched recursively).
    data_split_seed:
        RNG seed for the shuffle.
        Changing this seed produces a different split.
        ``data_split_seed`` is independent of the training RNG seed.
    val_ratio:
        Fraction of files to hold out for validation (must be in (0, 1)).

    Returns
    -------
    train_files, val_files:
        Two disjoint lists of ``.npz`` file paths.
    """
    files = _collect_npz_files(root)
    if len(files) < 2:
        raise ValueError(
            f"Expected at least two rollout files under {root}, found {len(files)}"
        )
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be strictly between 0 and 1, got {val_ratio}")
    rng = np.random.RandomState(data_split_seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    return files[n_val:], files[:n_val]
