import os
import random
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def load_rollouts_from_scenario(
    scenarios_dir: str
) -> Tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """
    Randomly select a rollout file (.npz) from `scenarios_dir` and return its observations and actions.

    Args:
        scenarios_dir: Path to a folder containing .npz rollouts saved with keys
                       'obs' (uint8 frames) and 'action' (float32 actions).

    Returns:
        obs: NDArray of shape (T, H, W, C), dtype=uint8
        actions: NDArray of shape (T, action_dim), dtype=float32
    """
    # 1. Find all .npz files
    files = [f for f in os.listdir(scenarios_dir) if f.endswith(".npz")]
    if not files:
        raise FileNotFoundError(f"No .npz files found in {scenarios_dir}")

    # 2. Pick one at random
    fname = random.choice(files)
    path = os.path.join(scenarios_dir, fname)

    # 3. Load and extract
    data = np.load(path)
    obs = data["obs"]        # e.g. shape (T, H, W, C), uint8
    actions = data["action"] # e.g. shape (T, action_dim), float32

    return obs, actions
