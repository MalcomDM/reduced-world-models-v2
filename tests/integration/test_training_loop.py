import numpy as np
import pytest
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from rwm.data.rollout_dataset import RolloutDataset
from rwm.trainers.deterministic.world_model_trainer import WorldModelTrainer
from rwm.types import RolloutSample



def create_dummy_rollout(dirpath: Path, name: str = "dummy"):
    """
    Create a single .npz rollout with:
      - T=20 timesteps
      - HxW=8x8, C=3
      - zero rewards (so the model can overfit to zero)
      - no 'done' flags
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    obs = np.random.randint(0, 255, size=(20, 8, 8, 3), dtype=np.uint8)
    action = np.zeros((20, 3), dtype=np.float32)
    reward = np.zeros(20, dtype=np.float32)
    done = np.zeros(20, dtype=bool)
    filepath = dirpath / f"{name}.npz"
    np.savez_compressed(filepath, obs=obs, action=action, reward=reward, done=done)
    return filepath


@pytest.fixture
def small_loader(tmp_path: Path) -> DataLoader[RolloutSample]:
    # Create two scenario directories, each with one dummy rollout
    dirs: List[Path] = []
    for i in range(2):
        d = tmp_path / f"scenario_{i}"
        create_dummy_rollout(d, name=f"r{i}")
        dirs.append(d)

    # Build a single DataLoader from both scenarios
    datasets = [
        RolloutDataset(d, sequence_len=5, image_size=8)
        for d in dirs
    ]
    full_ds = sum(datasets[1:], datasets[0])  # ConcatDataset
    loader = DataLoader(full_ds, batch_size=2, shuffle=False, drop_last=True)
    return loader


@pytest.mark.training
def test_rollout_dataset_and_loader_shapes(small_loader: DataLoader[RolloutSample]):
	batch = next(iter(small_loader))

	obs = batch["obs"]							# obs: (B, T, C, H, W)
	assert isinstance(obs, torch.Tensor)
	assert obs.ndim == 5 and obs.shape[2:] == (3, 8, 8)

	# action, reward, done
	act = batch["action"]
	rew = batch["reward"]
	done = batch["done"]
	assert act.shape == (2, 5, 3)
	assert rew.shape == (2, 5)
	assert done.shape == (2, 5)
    


@pytest.mark.training
def test_trainer_smoke_epoch_and_evaluate(small_loader: DataLoader[RolloutSample], tmp_path: Path):
    # Smoke-test one epoch and evaluation
    trainer = WorldModelTrainer(
        loader=small_loader,
        out_dir=tmp_path / "out",
        sequence_len=5,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        alpha=1.0,
        beta=0.1,
    )

    # train_one_epoch returns (loss, time)
    loss, elapsed = trainer.train_one_epoch()
    assert isinstance(loss, float) and loss >= 0.0
    assert isinstance(elapsed, float) and elapsed >= 0.0

    # evaluate returns both metrics
    metrics = trainer.evaluate()
    assert "mae_cum" in metrics and "mae_step" in metrics
    assert isinstance(metrics["mae_cum"], float)
    assert isinstance(metrics["mae_step"], float)


@pytest.mark.training
def test_overfit_single_batch(small_loader: DataLoader[RolloutSample], tmp_path: Path):
    """ Overfit to zero-reward data: two epochs on the same batch should reduce train_loss. """
    trainer = WorldModelTrainer(
        loader=small_loader,
        out_dir=tmp_path / "overfit",
        sequence_len=5,
        epochs=2,
        batch_size=2,
        lr=1e-2,   # higher LR for faster overfit
        alpha=1.0,
        beta=0.1,
    )

    loss1, _ = trainer.train_one_epoch()
    loss2, _ = trainer.train_one_epoch()
    assert loss2 <= loss1 + 1e-6, f"Expected loss to decrease or stay equal, got {loss1:.4f} → {loss2:.4f}"


@pytest.mark.training
def test_resource_usage_metrics(small_loader: DataLoader[RolloutSample], tmp_path: Path):
    """ Ensure we can read CPU and (if available) GPU memory stats during an epoch. """
    import psutil
    import torch

    # Before epoch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    proc = psutil.Process()

    trainer = WorldModelTrainer(
        loader=small_loader,
        out_dir=tmp_path / "res",
        sequence_len=5,
        epochs=1,
        batch_size=2,
    )
    trainer.train_one_epoch()

    # After epoch
    gpu_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    cpu_rss = proc.memory_info().rss
    assert isinstance(cpu_rss, int) and cpu_rss > 0
    if torch.cuda.is_available():
        assert isinstance(gpu_peak, int) and gpu_peak >= 0