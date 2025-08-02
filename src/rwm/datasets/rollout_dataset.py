import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from rwm.types import RolloutSample



class RolloutDataset(Dataset[RolloutSample]):
    def __init__(
        self,
        root_dir: Path,
        sequence_len: int = 16,
        image_size: int = 64,
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
        include_done: bool = False,
    ):
        self.sequence_len = sequence_len
        self.include_done = include_done
        self.transform: Callable[[Image.Image], Tensor] = transform or transforms.Compose([
            transforms.ToTensor(),  # (H, W, C) to (C, H, W), scaled to [0,1]
            transforms.Resize((image_size, image_size), antialias=True),
        ])

        self.samples: list[tuple[Path, int]] = []

        npz_files = list(root_dir.rglob("*.npz"))
        for path in npz_files:
            with np.load(path) as data:
                done = data["done"]
                T = len(done)
                for i in range(T - sequence_len + 1):
                    if not include_done and np.any(done[i:i+sequence_len]):
                        continue
                    self.samples.append((path, i))

        if not self.samples:
            raise RuntimeError(f"No valid sequences found in {root_dir}")


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int) -> RolloutSample:
        file_path, offset = self.samples[idx]
        with np.load(file_path) as data:
            obs_seq = data["obs"][offset : offset + self.sequence_len]
            act_seq = data["action"][offset : offset + self.sequence_len]
            rew_seq = data["reward"][offset : offset + self.sequence_len]
            done_seq = data["done"][offset : offset + self.sequence_len]

        obs_seq = [self.transform(Image.fromarray(frame)) for frame in obs_seq]
        obs_tensor = torch.stack(obs_seq)  # (T, C, H, W)

        return {
            "obs": obs_tensor,
            "action": torch.tensor(act_seq, dtype=torch.float32),
            "reward": torch.tensor(rew_seq, dtype=torch.float32),
            "done": torch.tensor(done_seq, dtype=torch.bool),
            "path": str(file_path),  # optional: for debugging
            "offset": offset,
        }