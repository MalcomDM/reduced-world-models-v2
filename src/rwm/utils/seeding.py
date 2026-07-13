"""Seed and reproducibility helpers.

Usage::

    from rwm.utils.seeding import set_seed
    set_seed(42)

    from rwm.utils.seeding import set_seed, SeedContext
    with SeedContext(42):
        ...

All functions record the seed and deterministic flag for later persistence
in run metadata.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


_CURRENT_SEED: Optional[int] = None
_CURRENT_DETERMINISTIC: bool = False


def get_current_seed() -> Optional[int]:
    """Return the seed last passed to ``set_seed``, or ``None``."""
    return _CURRENT_SEED


def get_deterministic_flag() -> bool:
    """Return whether deterministic mode has been requested."""
    return _CURRENT_DETERMINISTIC


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Parameters
    ----------
    seed:
        Integer seed.
    deterministic:
        If ``True``, prefer deterministic cuDNN behavior (may reduce
        performance). This does not guarantee that every PyTorch CUDA
        operation is deterministic.
    """
    global _CURRENT_SEED, _CURRENT_DETERMINISTIC
    _CURRENT_SEED = seed
    _CURRENT_DETERMINISTIC = deterministic

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


class SeedContext:
    """Context manager that temporarily sets a seed and restores RNG state
    on exit.

    Example::

        with SeedContext(99):
            x = torch.randn(3)  # reproducible
    """

    def __init__(self, seed: int, deterministic: bool = False):
        self._seed = seed
        self._deterministic = deterministic
        self._py_state: list = []
        self._np_state: Optional[tuple] = None
        self._torch_state: Optional[torch.Tensor] = None
        self._previous_seed: Optional[int] = None
        self._previous_deterministic: bool = False
        self._cudnn_deterministic: Optional[bool] = None
        self._cudnn_benchmark: Optional[bool] = None
        if torch.cuda.is_available():
            self._cuda_state: Optional[list[torch.Tensor]] = None
        else:
            self._cuda_state = None

    def __enter__(self) -> "SeedContext":
        global _CURRENT_SEED, _CURRENT_DETERMINISTIC
        self._previous_seed = _CURRENT_SEED
        self._previous_deterministic = _CURRENT_DETERMINISTIC
        self._py_state = random.getstate()
        self._np_state = np.random.get_state()
        self._torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            self._cuda_state = torch.cuda.get_rng_state_all()
            self._cudnn_deterministic = torch.backends.cudnn.deterministic
            self._cudnn_benchmark = torch.backends.cudnn.benchmark
        set_seed(self._seed, self._deterministic)
        return self

    def __exit__(self, *args: object) -> None:
        global _CURRENT_SEED, _CURRENT_DETERMINISTIC
        random.setstate(self._py_state)
        if self._np_state is not None:
            np.random.set_state(self._np_state)
        if self._torch_state is not None:
            torch.random.set_rng_state(self._torch_state)
        if self._cuda_state is not None:
            torch.cuda.set_rng_state_all(self._cuda_state)
        if self._cudnn_deterministic is not None:
            torch.backends.cudnn.deterministic = self._cudnn_deterministic
        if self._cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = self._cudnn_benchmark
        _CURRENT_SEED = self._previous_seed
        _CURRENT_DETERMINISTIC = self._previous_deterministic
