import pytest
import numpy as np
from typing import Tuple
from pathlib import Path

import torch

from rwm.config.config import ACTION_DIM
from rwm.utils.rollout_simulator import RolloutSimulator
from rwm.policies.controller_policy import ControllerPolicy
from rwm.models.rwm_deterministic.model import ReducedWorldModel
from rwm.models.controller.model import Controller


@pytest.fixture
def dummy_world_model():
    model = ReducedWorldModel()
    # stub generate_spatial_rep to return zeros of correct PRNN dimension
    def dummy_generate_spatial_rep(img_t: torch.Tensor) -> torch.Tensor:
        prnn_dim = model.world_rnn.rnn_cell.input_size - ACTION_DIM
        return torch.zeros(1, prnn_dim)
    model.generate_spatial_rep = dummy_generate_spatial_rep  # type: ignore[assignment]

    # stub forward: keep hidden/state unchanged, zero reward
    def dummy_forward(
        img: torch.Tensor,
        a_prev: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        force_keep_input: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = h_prev.shape[0]
        dummy_mask    = torch.zeros(B, 1)
        dummy_indices = torch.zeros(B, 1, dtype=torch.long)
        return h_prev, c_prev, torch.zeros(B, 1), dummy_mask, dummy_indices
    model.forward = dummy_forward  # type: ignore[assignment]

    return model

@pytest.fixture
def dummy_controller():
    return Controller()

def create_dummy_rollout(tmp_path: Path) -> str:
    obs = np.zeros((10, 64, 64, 3), dtype=np.uint8)
    actions = np.zeros((10, ACTION_DIM), dtype=np.float32)
    file_path = tmp_path / "rollout.npz"
    np.savez(file_path, obs=obs, action=actions)
    return str(tmp_path)


@pytest.mark.integration
def test_generate_rollouts(tmp_path: Path, dummy_world_model: ReducedWorldModel, dummy_controller: Controller):
    scenarios_dir = create_dummy_rollout(tmp_path)
    policy = ControllerPolicy(dummy_controller, noise_std=0.0)

    sim = RolloutSimulator(
        model=dummy_world_model,
        policy=policy,
        warmup_steps=2,
        rollout_len=3,
        device='cpu'
    )

    rollouts = sim.generate_rollouts(scenarios_dir, n=2)

    # Expect exactly 2 rollouts
    assert len(rollouts) == 2
    for latents, actions, rewards, cum in rollouts:
        # Each imagined rollout should have length == rollout_len
        assert len(latents) == 3
        assert len(actions) == 3
        assert len(rewards) == 3
        assert cum == sum(rewards)