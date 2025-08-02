import os, torch
import numpy as np
from typing import Any
from torch.utils.data import Dataset

from rwm.config.config import DATASET_DIR, SEQ_LEN

class WindowedDataset(Dataset[Any]):

    def __init__(self, attrs: tuple = ('obs', 'action', 'reward')):
        # 1) Cargar todos los episodios en memoria (igual que antes)
        self.episodes = []
        self.attrs = attrs
        for root, _, files in os.walk(DATASET_DIR):
            for fname in files:
                if not fname.endswith('.npz'):
                    continue
                path = os.path.join(root, fname)
                data = np.load(path)
                obs_np    = data[attrs[0]]    # (T_i, H, W, C)
                action_np = data[attrs[1]]    # (T_i, action_dim)
                reward_np = data[attrs[2]]    # (T_i,) o (T_i,1)

                T_i = obs_np.shape[0]
                if T_i < SEQ_LEN:
                    continue  # descartamos episodios más cortos que SEQ_LEN

                # Convertir a tensores
                obs_t = torch.from_numpy(obs_np.astype(np.float32)).permute(0, 3, 1, 2)  # (T_i, C, H, W)
                action_t = torch.from_numpy(action_np.astype(np.float32))               # (T_i, action_dim)
                reward_np = reward_np.astype(np.float32)
                if reward_np.ndim == 1:
                    reward_np = reward_np.reshape(-1, 1)
                reward_t = torch.from_numpy(reward_np)                                   # (T_i, 1)

                self.episodes.append((obs_t, action_t, reward_t))

        # 2) Construir índice global de ventanas: lista de (ep_idx, start_idx)
        self.index_map = []
        for ep_idx, (obs_t, action_t, reward_t) in enumerate(self.episodes):
            T_i = obs_t.shape[0]
            for start in range(0, T_i - SEQ_LEN + 1):
                self.index_map.append((ep_idx, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ep_idx, start = self.index_map[idx]
        obs_t, action_t, reward_t = self.episodes[ep_idx]

        end = start + SEQ_LEN
        obs_window    = obs_t[start:end]     # (SEQ_LEN, C, H, W)
        action_window = action_t[start:end]  # (SEQ_LEN, action_dim)
        reward_window = reward_t[start:end]  # (SEQ_LEN, 1)

        return obs_window, action_window, reward_window