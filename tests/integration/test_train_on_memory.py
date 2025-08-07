import numpy as np
import torch
import pytest
from pathlib import Path

from rwm.loops.train_controller import ControllerTrainer
from rwm.utils.behavior_memory import BehaviorMemory
from rwm.utils.rollout_simulator import RolloutSimulator
from rwm.models.rwm_deterministic.model import ReducedWorldModel
from rwm.models.controller.model import Controller
from rwm.policies.controller_policy import ControllerPolicy

@pytest.fixture
def small_memory(tmp_path: Path) -> BehaviorMemory:
    # Create a BehaviorMemory with one dummy situation
    mem_dir = tmp_path / "mem"
    bm = BehaviorMemory(str(mem_dir), max_size=5)

    # Build dummy obs/actions: warmup_steps=2 + rollout_len=3
    obs = np.zeros((5,64,64,3), dtype=np.uint8)
    acts = np.zeros((5,3), dtype=np.float32)
    h0 = torch.zeros(1, ReducedWorldModel().world_rnn.rnn_cell.hidden_size)

    # Add one entry with cum_reward=1.0
    bm.add(h0, 1.0, obs, acts)
    return bm

@pytest.fixture
def trainer(tmp_path: Path, small_memory: BehaviorMemory) -> ControllerTrainer:
    # Dummy world model & controller
    model = ReducedWorldModel()
    model.to("cpu").eval()
    ctrl = Controller().to("cpu")
    policy = ControllerPolicy(ctrl, noise_std=0.0)
    sim = RolloutSimulator(model, policy, warmup_steps=2, rollout_len=3, device="cpu")

    # Trainer with our small memory
    out_dir = str(tmp_path / "out")
    tr = ControllerTrainer(
        model=model, 
        model_ckpt="",  # skip loading
        out_dir=out_dir,
        controller=ctrl,
        scenarios_dirs=[], 
        warmup_steps=2, 
        rollout_len=3,
        device="cpu",
        simulator = sim
    )
    tr.memory = small_memory
    tr.memory_path = str(tmp_path / "out" / "mem.pkl")
    tr.optimizer = torch.optim.Adam(ctrl.parameters(), lr=1e-3)
    return tr


@pytest.mark.training
def test_train_on_memory_runs_and_updates(trainer: ControllerTrainer):
    # Grab initial weights
    init_params = [
        p.clone().detach() for p in trainer.controller.parameters()
    ]

    # Call train_on_memory
    loss = trainer.train_on_memory(batch_size=1)

    # Loss should be non-negative float
    assert isinstance(loss, float) and loss >= 0.0

    # At least one parameter should have changed
    updated = any(
        not torch.allclose(p_old, p_new, atol=1e-6)
        for p_old, p_new in zip(init_params, trainer.controller.parameters())
    )
    assert updated, "Controller parameters did not change during train_on_memory"
