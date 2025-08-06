import os
import random
import glob
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

def load_rollouts_from_scenario(
    scenarios_dir: str
) -> Tuple[NDArray[np.uint8], NDArray[np.float32]]:
	"""
	Recursively find all .npz rollouts under `scenarios_dir`, pick one at random,
	and return its 'obs' and 'action' arrays.
	"""
	# recursive glob for .npz
	pattern = os.path.join(scenarios_dir, '**', '*.npz')
	files = glob.glob(pattern, recursive=True)
	if not files:
		raise FileNotFoundError(f"No .npz files found in {scenarios_dir!r}")

	# pick a random rollout file
	filepath = random.choice(files)
	data = np.load(filepath)

	# ensure keys match
	if 'obs' not in data or 'action' not in data:
		raise KeyError(f"File {filepath!r} missing 'obs' or 'action' keys")

	obs: NDArray[np.uint8]     		= data['obs']        # shape (T,H,W,C), uint8
	actions: NDArray[np.float32] 	= data['action']     # shape (T,action_dim), float32
	return obs, actions
