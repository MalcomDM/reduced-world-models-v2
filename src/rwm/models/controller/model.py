import torch
import torch.nn as nn
from torch import Tensor

from rwm.config.config import ACTION_DIM, WRNN_HIDDEN_DIM


class Controller(nn.Module):
    def __init__(self,
		hidden_dim: int = WRNN_HIDDEN_DIM,
		action_dim: int = ACTION_DIM
    ) -> None:
        super().__init__()									# type: ignore[reportUnknownMemberType]
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            ActionSquash()
        )

    def forward(self, h: Tensor) -> Tensor:
        """
        Args:
            h: Tensor of shape (B, hidden_dim)
        Returns:
            actions: Tensor of shape (B, action_dim) in [-1,1]
        """
        return self.net(h)
    

class ActionSquash(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        steer = torch.tanh(x[:, 0:1])
        gas   = torch.sigmoid(x[:, 1:2])
        brake = torch.sigmoid(x[:, 2:3])
        return torch.cat([steer, gas, brake], dim=1)