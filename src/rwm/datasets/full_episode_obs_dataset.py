import os
import torch
import numpy as np
from typing import Any
from torch.utils.data import Dataset

from config.config import DATASET_DIR

                
class FullEpisodeObsDataset(Dataset[Any]):
    def __init__(self,
                 attr: str = 'obs'):
        self.episodes = []
        for root, _, files in os.walk(DATASET_DIR):
            for fname in files:
                if not fname.endswith('.npz'):
                    continue
                path = os.path.join(root, fname)
                data = np.load(path)[attr]
                ep = torch.from_numpy(data.astype(np.float32)).permute(0, 3, 1, 2) # type: ignore
                self.episodes.append(ep) # type: ignore


    def __len__(self):
        return len(self.episodes) # type: ignore


    def __getitem__(self, idx): # type: ignore
        return self.episodes[idx] # type: ignore
