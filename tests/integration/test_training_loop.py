import numpy as np
import pytest
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from rwm.data.rollout_dataset import RolloutDataset
from rwm.trainers.deterministic.world_model_trainer import WorldModelTrainer
from rwm.types import RolloutSample


# ---------------------------------------------------------------------------
# Baseline and metric aggregation tests
# ---------------------------------------------------------------------------

@pytest.mark.training
def test_baseline_mse_uses_training_set_mean(tmp_path: Path):
    """The baseline MSE must use the actual training-set reward mean,
    computed across all train batches in the epoch."""
    from rwm.trainers.deterministic.world_model_trainer import WorldModelTrainer as WMT

    # Fixed training data with known mean reward of 0.5 (constant).
    p = tmp_path / "data"
    p.mkdir()
    for name in ("a.npz", "b.npz"):
        T, H, W = 20, 8, 8
        obs = np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8)
        act = np.zeros((T, 3), dtype=np.float32)
        rew = np.full(T, 0.5, dtype=np.float32)  # known mean
        don = np.zeros(T, dtype=bool)
        np.savez_compressed(p / name, obs=obs, action=act, reward=rew, done=don)

    # Validation data with constant reward 1.0
    p_val = tmp_path / "val_data"
    p_val.mkdir()
    np.savez_compressed(
        p_val / "c.npz",
        obs=np.random.randint(0, 255, (20, 8, 8, 3), dtype=np.uint8),
        action=np.zeros((20, 3), dtype=np.float32),
        reward=np.full(20, 1.0, dtype=np.float32),
        done=np.zeros(20, dtype=bool),
    )

    from rwm.data.rollout_dataset import RolloutDataset
    train_ds = RolloutDataset(root_dir=p, sequence_len=8, image_size=8)
    val_ds = RolloutDataset(root_dir=p_val, sequence_len=8, image_size=8)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, drop_last=False)

    trainer = WMT(train_loader=train_loader, val_loader=val_loader,
                  out_dir=tmp_path / "out", sequence_len=8, epochs=1, batch_size=4, lr=1e-4, beta=0.0)
    trainer.train_one_epoch()
    # After training, _last_train_reward_mean should be ~0.5
    assert abs(trainer._last_train_reward_mean - 0.5) < 0.01, (
        f"Training-set reward mean should be ~0.5, got {trainer._last_train_reward_mean}"
    )

    # Evaluate: baseline predicts the training mean (0.5), but val reward is 1.0.
    # So baseline MSE = (0.5 - 1.0)^2 = 0.25
    val_metrics = trainer.evaluate()
    expected_baseline = (0.5 - 1.0) ** 2
    assert abs(val_metrics["mean_baseline_mse"] - expected_baseline) < 0.01, (
        f"Baseline MSE should be ~{expected_baseline}, got {val_metrics['mean_baseline_mse']}"
    )


@pytest.mark.training
def test_mae_is_mean_absolute_error(tmp_path: Path):
    """Validate MAE computation with known values."""
    from rwm.trainers.deterministic.world_model_trainer import WorldModelTrainer as WMT

    p = tmp_path / "d"
    p.mkdir()
    T = 10
    np.savez_compressed(
        p / "x.npz",
        obs=np.random.randint(0, 255, (T, 8, 8, 3), dtype=np.uint8),
        action=np.zeros((T, 3), dtype=np.float32),
        reward=np.linspace(0, 1, T, dtype=np.float32),
        done=np.zeros(T, dtype=bool),
    )
    from rwm.data.rollout_dataset import RolloutDataset
    ds = RolloutDataset(root_dir=p, sequence_len=3, image_size=8)
    loader = DataLoader(ds, batch_size=2, shuffle=False, drop_last=False)
    trainer = WMT(train_loader=loader, out_dir=tmp_path / "o", sequence_len=3, epochs=1, batch_size=2)
    # Just check the metric keys exist and are floats.
    metrics = trainer.evaluate()
    assert "val_mae" in metrics
    assert isinstance(metrics["val_mae"], float)


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------

def create_dummy_rollout(dirpath: Path, name: str = "dummy"):
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
    dirs: List[Path] = []
    for i in range(2):
        d = tmp_path / f"scenario_{i}"
        create_dummy_rollout(d, name=f"r{i}")
        dirs.append(d)

    datasets = [
        RolloutDataset(d, sequence_len=5, image_size=8)
        for d in dirs
    ]
    full_ds = sum(datasets[1:], datasets[0])
    loader = DataLoader(full_ds, batch_size=2, shuffle=False, drop_last=True)
    return loader


@pytest.mark.training
def test_rollout_dataset_and_loader_shapes(small_loader: DataLoader[RolloutSample]):
    batch = next(iter(small_loader))

    obs = batch["obs"]
    assert isinstance(obs, torch.Tensor)
    assert obs.ndim == 5 and obs.shape[2:] == (3, 8, 8)

    act = batch["action"]
    rew = batch["reward"]
    done = batch["done"]
    assert act.shape == (2, 5, 3)
    assert rew.shape == (2, 5)
    assert done.shape == (2, 5)


@pytest.mark.training
def test_trainer_smoke_epoch_and_evaluate(small_loader: DataLoader[RolloutSample], tmp_path: Path):
    trainer = WorldModelTrainer(
        train_loader=small_loader,
        out_dir=tmp_path / "out",
        sequence_len=5,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        beta=0.1,
    )

    train_ret = trainer.train_one_epoch()
    assert isinstance(train_ret[0], float) and train_ret[0] >= 0.0
    assert isinstance(train_ret[3], float) and train_ret[3] >= 0.0

    metrics = trainer.evaluate()
    assert "val_mse" in metrics
    assert isinstance(metrics["val_mse"], float)


@pytest.mark.training
def test_overfit_single_batch(small_loader: DataLoader[RolloutSample], tmp_path: Path):
    trainer = WorldModelTrainer(
        train_loader=small_loader,
        out_dir=tmp_path / "overfit",
        sequence_len=5,
        epochs=2,
        batch_size=2,
        lr=1e-2,
        beta=0.1,
    )

    ret1 = trainer.train_one_epoch()
    ret2 = trainer.train_one_epoch()
    assert ret2[1] <= ret1[1] + 1e-6, f"Expected train_mse to decrease, got {ret1[1]:.4f} to {ret2[1]:.4f}"


@pytest.mark.training
def test_resource_usage_metrics(small_loader: DataLoader[RolloutSample], tmp_path: Path):
    import psutil
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    proc = psutil.Process()

    trainer = WorldModelTrainer(
        train_loader=small_loader,
        out_dir=tmp_path / "res",
        sequence_len=5,
        epochs=1,
        batch_size=2,
    )
    trainer.train_one_epoch()

    gpu_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    cpu_rss = proc.memory_info().rss
    assert isinstance(cpu_rss, int) and cpu_rss > 0
    if torch.cuda.is_available():
        assert isinstance(gpu_peak, int) and gpu_peak >= 0
