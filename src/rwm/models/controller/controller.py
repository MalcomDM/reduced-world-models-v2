import torch
import torch.nn as nn

from app.models.reduced_world_model import ReducedWorldModel
from app.config import ACTION_DIM, WRNN_HIDDEN_DIM, OBSERVATIONAL_DROPOUT, SEQ_LEN
from app.datasets.windowed_dataset import WindowedDataset

class Controller(nn.Module):
    def __init__(self,
                 hidden_dim: int = WRNN_HIDDEN_DIM,
                 action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()   # asume acciones en [-1,1]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, hidden_dim)  → action: (B, action_dim)
        return self.net(h)