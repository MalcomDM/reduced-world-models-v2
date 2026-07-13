"""Deterministic probe set for measuring latent drift.

A probe set is a fixed batch of synthetic observations and actions that can
be passed through the model at different training stages to detect changes
in latent representations, reward predictions, or attention patterns.

Storage format: ``.npz`` with keys ``obs`` (N, H, W, C) uint8 and
``action`` (N, A) float32.  The default probe has N=8 samples with
H=W=64 and A=3.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from numpy.typing import NDArray

from rwm.config.config import ACTION_DIM


DEFAULT_PROBE_SIZE = 8
DEFAULT_IMAGE_SIZE = 64
DEFAULT_SEED = 0


def generate_probe_set(
    n: int = DEFAULT_PROBE_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = DEFAULT_SEED,
) -> Tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Generate a fixed deterministic probe set.

    Parameters
    ----------
    n:
        Number of probe samples.
    image_size:
        Height and width of probe images (square).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    obs:
        ``(n, H, W, 3)`` uint8 array of random pixel values.
    action:
        ``(n, A)`` float32 array within CarRacing's action bounds: steering
        in ``[-1, 1]`` and gas/brake in ``[0, 1]``.
    """
    rng = np.random.RandomState(seed)
    obs = rng.randint(0, 256, size=(n, image_size, image_size, 3), dtype=np.uint8)
    action = rng.uniform(0.0, 1.0, size=(n, ACTION_DIM)).astype(np.float32)
    action[:, 0] = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    return obs, action


def save_probe_set(
    path: Path,
    obs: NDArray[np.uint8],
    action: NDArray[np.float32],
) -> Path:
    """Save a probe set to a ``.npz`` file.

    Parameters
    ----------
    path:
        Destination path (``.npz`` extension added if missing).
    obs, action:
        Probe arrays from ``generate_probe_set()``.

    Returns
    -------
    path:
        The actual file path written.
    """
    path = path.with_suffix(".npz")
    np.savez_compressed(path, obs=obs, action=action)
    return path


def load_probe_set(path: Path) -> Tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Load a probe set from a ``.npz`` file.

    Returns
    -------
    (obs, action) tuple.
    """
    data = np.load(path.with_suffix(".npz"))
    return data["obs"], data["action"]


def make_default_probe(path: Optional[Path] = None) -> Tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Generate and optionally save the default probe set.

    The default probe has 8 samples, 64x64 images, seed 0, and serves as
    the canonical probe for drift measurement across all experiments.

    Parameters
    ----------
    path:
        If given, save the probe to this path.

    Returns
    -------
    (obs, action) tuple.
    """
    obs, action = generate_probe_set()
    if path is not None:
        save_probe_set(path, obs, action)
    return obs, action
