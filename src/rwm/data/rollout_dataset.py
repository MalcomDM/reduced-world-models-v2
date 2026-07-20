import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple

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
from rwm.data.burn_in_layout import (
    compute_burn_in_layout,
    build_loss_mask as _build_loss_mask,
    build_valid_step_mask as _build_valid_step_mask,
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
    recurrent_context: bool = False,
    burn_in_steps: int = 20,
) -> tuple["RolloutDataset", "RolloutDataset"]:
    """Build train and validation ``RolloutDataset`` instances with
    episode-safe splitting.

    Parameters
    ----------
    recurrent_context:
        If ``True``, load burn-in context frames before each window (SRU mode).
        When ``False``, load exact windows (causal mode, default).
    burn_in_steps:
        Number of burn-in context steps (only used when ``recurrent_context=True``).
    """
    train_files, val_files = episode_safe_train_val_split(
        root_dir, val_ratio=val_ratio, shuffle_seed=shuffle_seed,
    )
    train_ds = RolloutDataset.from_file_list(
        train_files, sequence_len=sequence_len,
        image_size=image_size, include_done=include_done,
        recurrent_context=recurrent_context,
        burn_in_steps=burn_in_steps,
    )
    val_ds = RolloutDataset.from_file_list(
        val_files, sequence_len=sequence_len,
        image_size=image_size, include_done=include_done,
        recurrent_context=recurrent_context,
        burn_in_steps=burn_in_steps,
    )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# RolloutDataset
# ---------------------------------------------------------------------------

class RolloutDataset(Dataset[RolloutSample]):
    """PyTorch Dataset over rollout ``.npz`` files.

    Two modes:

    * **Causal mode** (``recurrent_context=False``, default):
      Each sample is a sliding window of ``sequence_len`` timesteps from a
      single rollout file.  Windows containing ``done=True`` are excluded by
      default.

    * **SRU burn-in mode** (``recurrent_context=True``):
      Each sample contains a burn-in prefix followed by the target window.
      Total length is ``burn_in_steps + sequence_len``.  ``valid_step`` and
      ``loss_mask`` are returned alongside the data.

    When ``cache_dir`` is provided and contains pre-built frame tensors
    (via ``scripts/data/build_frame_cache.py``), observation loading uses the
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
        recurrent_context: bool = False,
        burn_in_steps: int = 20,
    ):
        self.sequence_len = sequence_len
        self.include_done = include_done
        self._recurrent_context = recurrent_context
        self._burn_in_steps = burn_in_steps if recurrent_context else 0
        self._image_size = image_size
        self._cache_manifest: Optional[Dict] = None
        self._cached_episodes: Dict[str, "np.ndarray"] = {}

        # Total loaded length in recurrent context mode.
        self._total_len = sequence_len + self._burn_in_steps if recurrent_context else sequence_len

        # Cache validation at init time
        if cache_dir is not None and str(cache_dir).strip():
            cd = Path(cache_dir)
            if not cd.exists():
                raise ValueError(
                    f"Cache directory {cd} does not exist. "
                    "Build the cache first:\n"
                    f"  python scripts/data/build_frame_cache.py --cache-dir {cd}"
                )
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
                if recurrent_context:
                    # In burn-in mode, the effective window starts at offset - burn_in.
                    # The first position where the full burn-in + target fits is
                    # burn_in_steps before the end of the episode.
                    end_limit = T - sequence_len
                    for i in range(end_limit + 1):
                        total_start = max(0, i - burn_in_steps)
                        total_end = min(T, i + sequence_len)
                        # Never cross done boundaries.
                        if not include_done and np.any(done[total_start:total_end]):
                            continue
                        self.samples.append((path, i))
                else:
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
        recurrent_context: bool = False,
        burn_in_steps: int = 20,
    ) -> "RolloutDataset":
        """Create a dataset from an explicit list of ``.npz`` file paths."""
        return cls(
            file_list=file_list,
            sequence_len=sequence_len,
            image_size=image_size,
            include_done=include_done,
            cache_dir=cache_dir,
            recurrent_context=recurrent_context,
            burn_in_steps=burn_in_steps,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RolloutSample:
        file_path, offset = self.samples[idx]

        if self._recurrent_context:
            return self._get_burn_in_item(file_path, offset)
        else:
            return self._get_standard_item(file_path, offset)

    def _get_standard_item(self, file_path: Path, offset: int) -> RolloutSample:
        """Standard causal mode: exact window of sequence_len."""
        with np.load(file_path) as data:
            act_seq = data["action"][offset: offset + self.sequence_len]
            rew_seq = data["reward"][offset: offset + self.sequence_len]
            done_seq = data["done"][offset: offset + self.sequence_len]
            if offset > 0:
                pred_action = data["action"][offset - 1]
            else:
                pred_action = np.zeros(act_seq.shape[1], dtype=np.float32)

        obs_tensor = self._load_obs(file_path, offset, self.sequence_len)

        return {
            "obs": obs_tensor,
            "action": torch.tensor(act_seq, dtype=torch.float32),
            "reward": torch.tensor(rew_seq, dtype=torch.float32),
            "done": torch.tensor(done_seq, dtype=torch.bool),
            "predecessor_action": torch.tensor(pred_action, dtype=torch.float32),
        }

    def _get_burn_in_item(self, file_path: Path, offset: int) -> RolloutSample:
        """SRU burn-in mode: fixed 36-position layout.

        Every sample has exactly ``burn_in_steps + sequence_len`` = 36 positions:
        - Positions [0, burn_in_steps) = 20: burn-in context (or left-padded zeros
          at episode start).
        - Positions [burn_in_steps, 36) = 16: target window.

        Early-episode missing context (offset < burn_in_steps) is left-padded
        with zeros.  Padded positions have ``valid_step=False`` (state unchanged)
        and ``loss_mask=False`` (no loss contribution).

        The target transition ``offset + j`` always occupies layout position
        ``burn_in_steps + j``, regardless of offset.
        """
        with np.load(file_path) as data:
            actions_all = data["action"]
            rewards_all = data["reward"]
            done_all = data["done"]

        total_len = self._burn_in_steps + self.sequence_len  # always 36
        real_start = max(0, offset - self._burn_in_steps)     # first source position loaded
        n_real_ctx = offset - real_start                       # available context steps
        n_real = n_real_ctx + self.sequence_len                 # total non-padding positions
        padding_before = self._burn_in_steps - n_real_ctx       # left-pad count (0 when fully available)

        # --- Load real (non-padding) data ---
        real_end = offset + self.sequence_len
        act_real = actions_all[real_start:real_end]
        rew_real = rewards_all[real_start:real_end]
        done_real = done_all[real_start:real_end]

        # Predecessor action for the FIRST non-padding position.
        if real_start > 0:
            pred_action_np = actions_all[real_start - 1]
        else:
            pred_action_np = np.zeros(actions_all.shape[1], dtype=np.float32)

        # --- Left-pad to fixed total_len ---
        def _left_pad_1d(arr: np.ndarray, pad: int) -> np.ndarray:
            if pad <= 0:
                return arr
            pad_shape = (pad,) + arr.shape[1:]
            return np.concatenate([np.zeros(pad_shape, dtype=arr.dtype), arr], axis=0)

        def _left_pad_obs(t: torch.Tensor, pad: int) -> torch.Tensor:
            if pad <= 0:
                return t
            pad_t = torch.zeros(pad, *t.shape[1:], dtype=t.dtype)
            return torch.cat([pad_t, t], dim=0)

        act_seq = _left_pad_1d(act_real, padding_before)
        rew_seq = _left_pad_1d(rew_real, padding_before)
        done_seq = _left_pad_1d(done_real, padding_before)
        obs_tensor = _left_pad_obs(self._load_obs(file_path, real_start, n_real), padding_before)

        # --- Masks ---
        vs = [False] * padding_before + [True] * n_real
        lm = [False] * self._burn_in_steps + [True] * self.sequence_len

        return {
            "obs": obs_tensor,                                    # (36, 3, 64, 64)
            "action": torch.tensor(act_seq, dtype=torch.float32), # (36, 3)
            "reward": torch.tensor(rew_seq, dtype=torch.float32), # (36,)
            "done": torch.tensor(done_seq, dtype=torch.bool),     # (36,)
            "predecessor_action": torch.tensor(pred_action_np, dtype=torch.float32),  # (3,)
            "valid_step": torch.tensor(vs, dtype=torch.bool),     # (36,)
            "loss_mask": torch.tensor(lm, dtype=torch.bool),      # (36,)
        }

    def _load_obs(self, file_path: Path, start: int, length: int) -> Tensor:
        """Load observation segment, using cache if available."""
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
            cached_len = cached.shape[0]
            actual_end = min(start + length, cached_len)
            n_avail = max(0, actual_end - start)
            obs_tensor = torch.from_numpy(
                cached[start:actual_end].copy()
            )
            if n_avail < length:
                pad = torch.zeros(
                    length - n_avail, *obs_tensor.shape[1:],
                    dtype=obs_tensor.dtype,
                )
                obs_tensor = torch.cat([pad, obs_tensor], dim=0)
            return obs_tensor
        else:
            with np.load(file_path) as data:
                obs_seq = data["obs"][start: start + length]
            obs_seq = [self.transform(Image.fromarray(frame)) for frame in obs_seq]
            return torch.stack(obs_seq)
