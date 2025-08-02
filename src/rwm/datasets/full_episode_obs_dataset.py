import os
import torch
import numpy as np
from torch.utils.data import Dataset

from src.config import DATASET_DIR

                
class FullEpisodeObsDataset(Dataset):
    def __init__(self,
                 attr: str = 'obs'):
        self.episodes = []
        for root, _, files in os.walk(DATASET_DIR):
            for fname in files:
                if not fname.endswith('.npz'):
                    continue
                path = os.path.join(root, fname)
                data = np.load(path)[attr]
                T = data.shape[0]
                ep = torch.from_numpy(data.astype(np.float32)).permute(0, 3, 1, 2)
                self.episodes.append(ep)


    def __len__(self):
        return len(self.episodes)


    def __getitem__(self, idx):
        return self.episodes[idx]
