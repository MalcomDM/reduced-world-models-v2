"""Sequential episode dataset for SRU truncated-BPTT training.

Each batch contains chunks from ``batch_size`` independent episode streams,
each moving sequentially through its episode.  When a stream finishes an
episode it picks the next unused episode and resets its SRU state.

Every real source transition is processed exactly once per epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
from torchvision import transforms

from rwm.data.cache_utils import (
    load_manifest as _load_cache_manifest,
    verify_cache_entry as _verify_cache_entry,
)


def _pad_to_length(arr: np.ndarray, length: int) -> np.ndarray:
    if len(arr) >= length:
        return arr[:length]
    pad_shape = (length - len(arr),) + arr.shape[1:]
    return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)


class MultiStreamSequentialDataset(IterableDataset):
    """Iterable dataset yielding batches of sequential SRU chunks.

    Maintains ``batch_size`` independent episode streams.  Each stream
    processes one episode sequentially in contiguous non-overlapping chunks
    of ``chunk_len``.  When a stream exhausts its current episode, it
    advances to the next unused episode and its next chunk carries
    ``episode_start=True`` (signal to reset SRU state).

    Every real source transition is processed exactly once per complete pass
    through all episodes.

    Parameters
    ----------
    file_list:
        List of ``.npz`` episode files.
    chunk_len:
        Contiguous steps per chunk (tbptt_steps).
    batch_size:
        Number of independent episode streams.
    image_size:
        Resize each frame to this square size.
    cache_dir:
        Optional path to pre-built frame cache.
    """

    def __init__(
        self,
        file_list: List[Path],
        chunk_len: int = 16,
        batch_size: int = 8,
        image_size: int = 64,
        cache_dir: Optional[Path] = None,
    ):
        super().__init__()
        self.chunk_len = chunk_len
        self.batch_size = batch_size
        self._image_size = image_size
        self._files = file_list[:]
        self._cache_manifest: Optional[Dict] = None
        self._cached_episodes: Dict[str, np.ndarray] = {}
        self._episode_cache: Dict[Path, Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}

        if cache_dir is not None and str(cache_dir).strip():
            cd = Path(cache_dir)
            if not cd.exists():
                raise ValueError(f"Cache directory {cd} does not exist.")
            self._cache_manifest = _load_cache_manifest(cd, image_size=image_size)
            self._cache_dir = cd
        else:
            self._cache_dir = None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), antialias=True),
        ])

        if not file_list:
            raise RuntimeError("Empty file list")

        # Pre-load episode metadata.
        self._ep_actions: Dict[Path, np.ndarray] = {}
        self._ep_rewards: Dict[Path, np.ndarray] = {}
        self._ep_dones: Dict[Path, np.ndarray] = {}
        self._ep_lengths: Dict[Path, int] = {}

        for path in file_list:
            with np.load(path) as data:
                self._ep_actions[path] = data["action"]
                self._ep_rewards[path] = data["reward"]
                self._ep_dones[path] = data["done"]
                self._ep_lengths[path] = len(data["done"])

    def __iter__(self) -> Iterator[dict]:
        """Yield batches sequentially.

        Each yielded dict has key ``batch`` containing a list of ``B``
        per-stream dicts with keys:
            obs, action, reward, done, predecessor_action,
            valid_step, loss_mask, file_path, chunk_start, episode_start

        The outer dict also has a top-level ``valid_step`` and ``loss_mask``
        as stacked ``(B, T)`` bool tensors for convenience.
        """
        # Build per-stream state: each stream runs independently.
        # We use a round-robin through episodes: each stream starts with a
        # distinct episode, then picks the next unused one when done.
        num_ep = len(self._files)
        stream_ep_idx: List[int] = list(range(min(self.batch_size, num_ep)))
        stream_pos: List[int] = [0] * self.batch_size
        stream_files: List[Path] = [self._files[i] if i < num_ep else self._files[-1] for i in stream_ep_idx]
        used_ep_set: set[int] = set(stream_ep_idx)
        next_ep_idx = len(stream_ep_idx)

        while True:
            batch_items = []
            any_active = False
            for s in range(self.batch_size):
                fpath = stream_files[s]
                ep_len = self._ep_lengths[fpath]

                if stream_pos[s] >= ep_len:
                    # This stream needs a new episode.
                    if next_ep_idx < num_ep:
                        # Assign next unused episode.
                        stream_files[s] = self._files[next_ep_idx]
                        stream_pos[s] = 0
                        stream_ep_idx[s] = next_ep_idx
                        used_ep_set.add(next_ep_idx)
                        next_ep_idx += 1
                        fpath = stream_files[s]
                        ep_len = self._ep_lengths[fpath]
                    else:
                        # No more episodes — yield a padding-only batch item.
                        batch_items.append(None)
                        continue

                # After possibly picking a new episode, check if this stream is done.
                if stream_pos[s] >= self._ep_lengths[fpath]:
                    # Exhausted all episodes.
                    batch_items.append(None)
                    continue

                cs = stream_pos[s]
                cl = min(self.chunk_len, ep_len - cs)
                ep_start = (cs == 0)

                any_active = True

                # Load data.
                act_real = self._ep_actions[fpath][cs:cs + cl]
                rew_real = self._ep_rewards[fpath][cs:cs + cl]
                done_real = self._ep_dones[fpath][cs:cs + cl]

                act_seq = _pad_to_length(act_real, self.chunk_len)
                rew_seq = _pad_to_length(rew_real, self.chunk_len)
                done_seq = _pad_to_length(done_real, self.chunk_len)

                pred = self._ep_actions[fpath][cs - 1] if cs > 0 else \
                    np.zeros(act_real.shape[1], dtype=np.float32)

                # Observations.
                obs_t = self._load_obs(fpath, cs, cl)
                if obs_t.shape[0] < self.chunk_len:
                    pad = torch.zeros(
                        self.chunk_len - obs_t.shape[0], *obs_t.shape[1:],
                        dtype=obs_t.dtype,
                    )
                    obs_t = torch.cat([obs_t, pad], dim=0)

                vs = [True] * cl + [False] * (self.chunk_len - cl)
                lm = [True] * cl + [False] * (self.chunk_len - cl)

                batch_items.append({
                    "obs": obs_t,
                    "action": torch.tensor(act_seq, dtype=torch.float32),
                    "reward": torch.tensor(rew_seq, dtype=torch.float32),
                    "done": torch.tensor(done_seq, dtype=torch.bool),
                    "predecessor_action": torch.tensor(pred, dtype=torch.float32),
                    "valid_step": torch.tensor(vs, dtype=torch.bool),
                    "loss_mask": torch.tensor(lm, dtype=torch.bool),
                    "file_path": fpath,
                    "chunk_start": cs,
                    "episode_start": ep_start,
                })

                stream_pos[s] += self.chunk_len

            if not any_active:
                break

            # Stack batch tensors.
            obs_l, act_l, rew_l, done_l, pred_l = [], [], [], [], []
            vs_l, lm_l = [], []
            ep_start_l = []
            for item in batch_items:
                if item is None:
                    # Padding batch slot: zeros, valid_step=False, loss_mask=False
                    zero_obs = torch.zeros(self.chunk_len, 3, self._image_size, self._image_size)
                    zero_act = torch.zeros(self.chunk_len, 3)
                    zero_rew = torch.zeros(self.chunk_len)
                    zero_done = torch.zeros(self.chunk_len, dtype=torch.bool)
                    zero_pred = torch.zeros(3)
                    zero_vs = torch.zeros(self.chunk_len, dtype=torch.bool)
                    zero_lm = torch.zeros(self.chunk_len, dtype=torch.bool)
                    obs_l.append(zero_obs)
                    act_l.append(zero_act)
                    rew_l.append(zero_rew)
                    done_l.append(zero_done)
                    pred_l.append(zero_pred)
                    vs_l.append(zero_vs)
                    lm_l.append(zero_lm)
                    ep_start_l.append(False)
                else:
                    obs_l.append(item["obs"])
                    act_l.append(item["action"])
                    rew_l.append(item["reward"])
                    done_l.append(item["done"])
                    pred_l.append(item["predecessor_action"])
                    vs_l.append(item["valid_step"])
                    lm_l.append(item["loss_mask"])
                    ep_start_l.append(item["episode_start"])

            yield {
                "obs": torch.stack(obs_l, dim=0),
                "action": torch.stack(act_l, dim=0),
                "reward": torch.stack(rew_l, dim=0),
                "done": torch.stack(done_l, dim=0),
                "predecessor_action": torch.stack(pred_l, dim=0),
                "valid_step": torch.stack(vs_l, dim=0),
                "loss_mask": torch.stack(lm_l, dim=0),
                "episode_start": torch.tensor(ep_start_l, dtype=torch.bool),
                "file_path": [item["file_path"] if item is not None else Path("") for item in batch_items],
                "chunk_start": [item["chunk_start"] if item is not None else -1 for item in batch_items],
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
