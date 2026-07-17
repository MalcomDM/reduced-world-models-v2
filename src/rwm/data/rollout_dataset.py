import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Callable, List, Dict

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from torchvision import transforms			# type: ignore

from rwm.types import RolloutSample
from rwm.data.cache_utils import (
    load_manifest as _load_cache_manifest,
    verify_cache_entry as _verify_cache_entry,
    _CACHE_SCHEMA_VERSION,
    _TRANSFORM_SPEC,
    _DEFAULT_IMAGE_SIZE,
)


def _is_evaluation_file(path: Path) -> bool:
    """Check if an ``.npz`` file is evaluation-only data.

    Returns ``True`` if the file has a matching ``.episode.json`` or
    ``.branch.json`` sidecar.  Training code must not load such files.
    """
    if path.with_suffix(".episode.json").exists():
        return True
    if path.with_suffix(".branch.json").exists():
        return True
    return False


# ---------------------------------------------------------------------------
# Episode-safe split helpers
# ---------------------------------------------------------------------------

def _collect_npz_files(root_dir: Path) -> List[Path]:
    """Return all ``.npz`` rollout files under ``root_dir`` recursively."""
    return sorted(root_dir.rglob("*.npz"))


def episode_safe_train_val_split(
    root_dir: Path,
    val_ratio: float = 0.2,
    shuffle_seed: int = 42,
) -> tuple[List[Path], List[Path]]:
    """Split rollout files by **episode** (file), never by window.

    This guarantees that windows from the same episode never appear in both
    train and validation sets, preventing temporal leakage.

    Parameters
    ----------
    root_dir:
        Directory containing ``.npz`` rollout files (searched recursively).
    val_ratio:
        Fraction of files to hold out for validation.
    shuffle_seed:
        Seed for reproducible file-level shuffle before splitting.

    Returns
    -------
    train_files, val_files:
        Two disjoint lists of ``.npz`` file paths.
    """
    files = _collect_npz_files(root_dir)
    if len(files) < 2:
        raise ValueError(
            "Episode-safe train/validation splitting requires at least two "
            f"rollout files; found {len(files)} in {root_dir}."
        )
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between 0 and 1.")
    rng = np.random.RandomState(shuffle_seed)
    rng.shuffle(files)
    n_val = min(len(files) - 1, max(1, int(len(files) * val_ratio)))
    return files[n_val:], files[:n_val]


def build_train_val_datasets(
    root_dir: Path,
    sequence_len: int = 16,
    image_size: int = 64,
    val_ratio: float = 0.2,
    shuffle_seed: int = 42,
    include_done: bool = False,
) -> tuple["RolloutDataset", "RolloutDataset"]:
    """Build train and validation ``RolloutDataset`` instances with
    episode-safe splitting.

    Parameters
    ----------
    root_dir:
        Directory of rollout ``.npz`` files.
    sequence_len, image_size:
        Passed to each ``RolloutDataset``.
    val_ratio:
        Fraction of files to hold out for validation.
    shuffle_seed:
        Seed for reproducible file-level shuffle.

    Returns
    -------
    train_dataset, val_dataset
    """
    train_files, val_files = episode_safe_train_val_split(
        root_dir, val_ratio=val_ratio, shuffle_seed=shuffle_seed,
    )
    train_ds = RolloutDataset.from_file_list(
        train_files, sequence_len=sequence_len,
        image_size=image_size, include_done=include_done,
    )
    val_ds = RolloutDataset.from_file_list(
        val_files, sequence_len=sequence_len,
        image_size=image_size, include_done=include_done,
    )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# RolloutDataset
# ---------------------------------------------------------------------------

class RolloutDataset(Dataset[RolloutSample]):
	"""PyTorch Dataset over rollout ``.npz`` files.

	Each sample is a sliding window of ``sequence_len`` timesteps from a
	single rollout file. Windows containing ``done=True`` are excluded by
	default.

	When ``cache_dir`` is provided and contains pre-built frame tensors
	(via ``scripts/build_frame_cache.py``), observation loading uses the
	cache instead of decompressing NPZ + PIL transforms.  Action, reward,
	done, and predecessor_action are always read from the source NPZ.

	Transition semantics (per index ``t``):
	    obs[t]    = s_t           observation before action
	    action[t] = a_t           action selected from s_t
	    reward[t] = r_{t+1}       reward from env.step(a_t)
	    done[t]   = terminal_{t+1} whether s_{t+1} is terminal
	"""

	def __init__(
		self,
		root_dir: Optional[Path] = None,
		sequence_len: int = 16,
		image_size: int = 64,
		transform: Optional[Callable[[Image.Image], Tensor]] = None,
		include_done: bool = False,
		file_list: Optional[List[Path]] = None,
		cache_dir: Optional[Path] = None,
	):
		self.sequence_len = sequence_len
		self.include_done = include_done
		self._image_size = image_size
		self._cache_manifest: Optional[Dict] = None
		self._cached_episodes: Dict[str, "np.ndarray"] = {}

		# Cache validation at init time
		if cache_dir is not None and str(cache_dir).strip():
			cd = Path(cache_dir)
			if not cd.exists():
				raise ValueError(
					f"Cache directory {cd} does not exist. "
					"Build the cache first:\n"
					f"  python scripts/build_frame_cache.py --cache-dir {cd}"
				)
			# Validate manifest against requested image_size and default transform spec
			self._cache_manifest = _load_cache_manifest(
				cd, image_size=image_size, transform_spec=_TRANSFORM_SPEC,
			)
			if transform is not None:
				raise ValueError(
					"A custom Dataset transform cannot be used with a frame cache. "
					"The cache was built for the default transform "
					f"({_TRANSFORM_SPEC}, image_size={image_size}). "
					"Either omit the custom transform, or set cache_dir to None/empty."
				)
			self._cache_dir = cd
		else:
			self._cache_dir = None

		if transform is None:
			transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Resize((image_size, image_size), antialias=True),
			])
		self.transform: Callable[[Image.Image], Tensor] = transform

		self.samples: list[tuple[Path, int]] = []

		if file_list is not None:
			npz_files = file_list
		elif root_dir is not None:
			npz_files = list(root_dir.rglob("*.npz"))
		else:
			raise ValueError("Provide either root_dir or file_list")

		for path in npz_files:
			if _is_evaluation_file(path):
				continue
			with np.load(path) as data:
				done = data["done"]
				T = len(done)
				for i in range(T - sequence_len + 1):
					if not include_done and np.any(done[i:i+sequence_len]):
						continue
					self.samples.append((path, i))

		if not self.samples:
			raise RuntimeError(
				"No valid sequences found — ensure .npz files exist "
				"and contain windows without done=True"
			)


	@classmethod
	def from_file_list(
		cls,
		file_list: List[Path],
		sequence_len: int = 16,
		image_size: int = 64,
		include_done: bool = False,
		cache_dir: Optional[Path] = None,
	) -> "RolloutDataset":
		"""Create a dataset from an explicit list of ``.npz`` file paths."""
		return cls(
			file_list=file_list,
			sequence_len=sequence_len,
			image_size=image_size,
			include_done=include_done,
			cache_dir=cache_dir,
		)


	def __len__(self) -> int:
		return len(self.samples)


	def __getitem__(self, idx: int) -> RolloutSample:
		file_path, offset = self.samples[idx]
		with np.load(file_path) as data:
			act_seq = data["action"][offset : offset + self.sequence_len]
			rew_seq = data["reward"][offset : offset + self.sequence_len]
			done_seq = data["done"][offset : offset + self.sequence_len]
			if offset > 0:
				pred_action = data["action"][offset - 1]
			else:
				pred_action = np.zeros(act_seq.shape[1], dtype=np.float32)

		# Observations: cached path or PIL transform
		if self._cache_dir is not None:
			fs = str(file_path.resolve())
			if fs not in self._cached_episodes:
				cache_path = _verify_cache_entry(
					self._cache_dir, file_path,
					self._cache_manifest or {},
					self._image_size,
				)
				cached = np.load(cache_path, mmap_mode="r")
				self._cached_episodes[fs] = cached
			else:
				cached = self._cached_episodes[fs]
			obs_tensor = torch.from_numpy(
				cached[offset : offset + self.sequence_len].copy()
			)  # (T, 3, H, W)
		else:
			with np.load(file_path) as data:
				obs_seq = data["obs"][offset : offset + self.sequence_len]
			obs_seq = [self.transform(Image.fromarray(frame)) for frame in obs_seq]
			obs_tensor = torch.stack(obs_seq)

		return {
			"obs": obs_tensor,
			"action": torch.tensor(act_seq, dtype=torch.float32),
			"reward": torch.tensor(rew_seq, dtype=torch.float32),
			"done": torch.tensor(done_seq, dtype=torch.bool),
			"predecessor_action": torch.tensor(pred_action, dtype=torch.float32),
		}
