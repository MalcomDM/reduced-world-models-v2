"""Macroblock dataset for SRU random-macroblock TBPTT training.

Each sample is one macroblock: ``burn_in_steps`` context steps +
``macroblock_target_steps`` target steps (partitioned into
``macroblock_target_steps // tbptt_steps`` contiguous TBPTT chunks).
The dataset is map-style so normal shuffled batching works.

Default layout: 20 burn-in + 64 target = 84 total positions
(can be overridden via ``macroblock_target_steps=96`` etc.).
Left-padding at early-episode offsets; right-padding for short
final episode segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from rwm.data.cache_utils import (
    load_manifest as _load_cache_manifest,
    verify_cache_entry as _verify_cache_entry,
)
from rwm.data.burn_in_layout import compute_burn_in_layout


@dataclass(frozen=True)
class MacroblockSample:
    """Metadata for one macroblock sample."""
    file_path: Path
    target_start: int          # source offset where the target begins
    target_len: int            # real target steps (may be short at episode end)
    episode_len: int


def compute_macroblocks(
    episode_len: int,
    macroblock_target_steps: int,
    file_path: Path,
) -> List[MacroblockSample]:
    """Partition an episode into non-overlapping macroblock target regions.

    Targets are [0, M), [M, 2M), ... where M = macroblock_target_steps.
    The final target may be shorter.  Each macroblock loads its own
    preceding burn-in context when the sample is constructed.
    """
    samples: List[MacroblockSample] = []
    start = 0
    while start < episode_len:
        t_len = min(macroblock_target_steps, episode_len - start)
        samples.append(MacroblockSample(
            file_path=file_path,
            target_start=start,
            target_len=t_len,
            episode_len=episode_len,
        ))
        start += macroblock_target_steps
    return samples


def _pad_to_length(arr: np.ndarray, length: int) -> np.ndarray:
    if len(arr) >= length:
        return arr[:length]
    pad_shape = (length - len(arr),) + arr.shape[1:]
    return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)


class MacroblockDataset(Dataset):
    """Map-style dataset yielding fixed-size macroblock samples.

    Each sample has total length ``burn_in_steps + macroblock_target_steps``
    (default 20 + 64 = 84).  Fields match the usual ``RolloutSample`` dict
    plus ``valid_step``, ``burn_in_mask``, ``loss_mask``, and source offset
    metadata.

    Parameters
    ----------
    file_list:
        List of ``.npz`` episode files.
    burn_in_steps:
        Context steps before each macroblock target (default 20).
    macroblock_target_steps:
        Target steps per macroblock (default 64, must be divisible by tbptt_steps).
    image_size:
        Resize each frame to this square size.
    cache_dir:
        Optional pre-built frame cache.
    """

    def __init__(
        self,
        file_list: List[Path],
        burn_in_steps: int = 20,
        macroblock_target_steps: int = 64,
        image_size: int = 64,
        cache_dir: Optional[Path] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.burn_in_steps = burn_in_steps
        self.macroblock_target_steps = macroblock_target_steps
        self.total_len = burn_in_steps + macroblock_target_steps
        self._image_size = image_size
        self._cache_manifest: Optional[Dict] = None
        self._cached_episodes: Dict[str, np.ndarray] = {}

        if cache_dir is not None and str(cache_dir).strip():
            cd = Path(cache_dir)
            if not cd.exists():
                raise ValueError(f"Cache directory {cd} does not exist.")
            self._cache_manifest = _load_cache_manifest(cd, image_size=image_size)
            self._cache_dir = cd
        else:
            self._cache_dir = None

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size), antialias=True),
            ])
        self.transform = transform

        # Build all macroblock samples.
        self.samples: List[MacroblockSample] = []
        self._ep_actions: Dict[Path, np.ndarray] = {}
        self._ep_rewards: Dict[Path, np.ndarray] = {}
        self._ep_dones: Dict[Path, np.ndarray] = {}

        for path in file_list:
            with np.load(path) as data:
                self._ep_actions[path] = data["action"]
                self._ep_rewards[path] = data["reward"]
                self._ep_dones[path] = data["done"]
                ep_len = len(data["done"])
            self.samples.extend(compute_macroblocks(ep_len, macroblock_target_steps, path))

        if not self.samples:
            raise RuntimeError("No macroblock samples found")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        mb = self.samples[idx]
        fpath = mb.file_path
        target_start = mb.target_start
        target_len = mb.target_len
        burn_in = self.burn_in_steps
        total = self.total_len

        actions_all = self._ep_actions[fpath]
        rewards_all = self._ep_rewards[fpath]
        dones_all = self._ep_dones[fpath]

        # Burn-in context: up to burn_in steps before target_start.
        ctx_start = max(0, target_start - burn_in)
        n_ctx = target_start - ctx_start  # available context (0 when offset=0)

        # Load real data: context + target.
        real_start = ctx_start
        real_end = target_start + target_len
        act_real = actions_all[real_start:real_end]
        rew_real = rewards_all[real_start:real_end]
        done_real = dones_all[real_start:real_end]
        n_real = len(act_real)

        # Predecessor action for the first real (non-padded) position.
        if real_start > 0:
            pred_action = actions_all[real_start - 1]
        else:
            pred_action = np.zeros(actions_all.shape[1], dtype=np.float32)

        # Left-pad to total length.
        padding_before = burn_in - n_ctx  # >0 when target_start < burn_in
        n_total = padding_before + n_real
        # Right-pad final macroblock if target is shorter.
        # Since the layout is fixed burn_in + macroblock_target_steps,
        # the loaded real data may be less than total only when the
        # episode ends before total.
        n_avail = min(n_total, total)

        def _left_right_pad(arr: np.ndarray, left: int, right: int) -> np.ndarray:
            if left > 0 or right > 0:
                ps = (left,) + arr.shape[1:]
                return np.concatenate([np.zeros(ps, dtype=arr.dtype), arr[:n_avail - left],
                                       np.zeros((right,) + arr.shape[1:], dtype=arr.dtype)], axis=0)
            return arr[:n_avail]

        # How many total real positions we actually have.
        avail_before_target = burn_in - padding_before  # real context steps
        avail_target = min(target_len, self.macroblock_target_steps)
        n_avail_total = padding_before + avail_before_target + avail_target
        right_pad = total - n_avail_total

        act_seq = _left_right_pad(act_real, padding_before, right_pad)
        rew_seq = _left_right_pad(rew_real, padding_before, right_pad)
        done_seq = _left_right_pad(done_real, padding_before, right_pad)

        # Observations.
        obs_t = self._load_obs(fpath, real_start, n_real)
        if padding_before > 0:
            pad_obs = torch.zeros(padding_before, *obs_t.shape[1:], dtype=obs_t.dtype)
            obs_t = torch.cat([pad_obs, obs_t], dim=0)
        if obs_t.shape[0] < total:
            pad_obs2 = torch.zeros(total - obs_t.shape[0], *obs_t.shape[1:], dtype=obs_t.dtype)
            obs_t = torch.cat([obs_t, pad_obs2], dim=0)

        # Masks.
        valid_step = [False] * padding_before + [True] * (n_avail_total - padding_before) \
                     + [False] * right_pad
        burn_in_mask = [False] * padding_before + [True] * avail_before_target \
                       + [False] * (avail_target + right_pad)
        loss_mask = [False] * padding_before + [False] * avail_before_target \
                    + [True] * avail_target + [False] * right_pad

        return {
            "obs": obs_t,                                               # (total_len, 3, H, W)
            "action": torch.tensor(act_seq, dtype=torch.float32),       # (total_len, 3)
            "reward": torch.tensor(rew_seq, dtype=torch.float32),       # (total_len,)
            "done": torch.tensor(done_seq, dtype=torch.bool),           # (total_len,)
            "predecessor_action": torch.tensor(pred_action, dtype=torch.float32),  # (3,)
            "valid_step": torch.tensor(valid_step, dtype=torch.bool),   # (total_len,)
            "burn_in_mask": torch.tensor(burn_in_mask, dtype=torch.bool),  # (total_len,)
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),     # (total_len,)
            "target_start": target_start,
            "target_len": target_len,
        }

    def _load_obs(self, file_path: Path, start: int, length: int) -> Tensor:
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
            actual = min(start + length, cached.shape[0])
            n = max(0, actual - start)
            t = torch.from_numpy(cached[start:actual].copy())
            if n < length:
                pad = torch.zeros(length - n, *t.shape[1:], dtype=t.dtype)
                t = torch.cat([t, pad], dim=0)
            return t
        else:
            with np.load(file_path) as data:
                obs_seq = data["obs"][start: start + length]
            obs_seq = [self.transform(Image.fromarray(frame)) for frame in obs_seq]
            return torch.stack(obs_seq)
