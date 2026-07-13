import pytest
import numpy as np
from pathlib import Path

from rwm.data.rollout_dataset import (
    RolloutDataset,
    episode_safe_train_val_split,
    build_train_val_datasets,
)


@pytest.mark.dataset
def test_rollout_dataset_loads(tmp_path: Path):
    # Create minimal fake .npz file
    path = tmp_path / "test" / "sample.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(path,
        obs=np.zeros((20, 64, 64, 3), dtype=np.uint8),
        action=np.zeros((20, 3), dtype=np.float32),
        reward=np.zeros((20,), dtype=np.float32),
        done=np.zeros((20,), dtype=bool),
    )

    ds = RolloutDataset(root_dir=tmp_path, sequence_len=8)
    assert len(ds) == 13  # 20 - 8 + 1
    sample = ds[0]
    assert sample["obs"].shape == (8, 3, 64, 64)
    assert sample["action"].shape == (8, 3)


@pytest.mark.dataset
def test_done_filtering(tmp_path: Path):
    path = tmp_path / "test.npz"
    done = np.zeros(20, dtype=bool)
    done[5] = True
    np.savez_compressed(path,
        obs=np.zeros((20, 64, 64, 3), dtype=np.uint8),
        action=np.zeros((20, 3), dtype=np.float32),
        reward=np.zeros((20,), dtype=np.float32),
        done=done,
    )

    ds = RolloutDataset(root_dir=tmp_path, sequence_len=6, include_done=False)
    for _, offset in ds.samples:
        assert not done[offset : offset + 6].any()


@pytest.mark.dataset
def test_episode_safe_split_returns_disjoint_files(tmp_path: Path):
    """Train/val split should partition by file, never interleave."""
    for i in range(5):
        p = tmp_path / f"rollout_{i}.npz"
        np.savez_compressed(p,
            obs=np.zeros((20, 64, 64, 3), dtype=np.uint8),
            action=np.zeros((20, 3), dtype=np.float32),
            reward=np.zeros((20,), dtype=np.float32),
            done=np.zeros((20,), dtype=bool),
        )

    train_files, val_files = episode_safe_train_val_split(
        tmp_path, val_ratio=0.4, shuffle_seed=42,
    )

    # No file should appear in both splits
    train_set = set(train_files)
    val_set = set(val_files)
    assert train_set.isdisjoint(val_set), "Files overlap between train and val"

    # Total files should equal original count
    assert len(train_files) + len(val_files) == 5

    # Default val_ratio should leave at least 1 file in val
    assert len(val_files) >= 1


@pytest.mark.dataset
def test_build_train_val_datasets_respects_episodes(tmp_path: Path):
    """Two datasets built from an episode-safe split must not share
    windows derived from the same .npz file."""
    for i in range(4):
        p = tmp_path / f"run_{i}.npz"
        np.savez_compressed(p,
            obs=np.zeros((20, 64, 64, 3), dtype=np.uint8),
            action=np.zeros((20, 3), dtype=np.float32),
            reward=np.zeros((20,), dtype=np.float32),
            done=np.zeros((20,), dtype=bool),
        )

    train_ds, val_ds = build_train_val_datasets(
        tmp_path, sequence_len=5, val_ratio=0.25, shuffle_seed=0,
    )

    # Collect which files are referenced by each dataset
    train_files = {s[0] for s in train_ds.samples}
    val_files = {s[0] for s in val_ds.samples}
    assert train_files.isdisjoint(val_files), (
        "A rollout file appears in both train and validation datasets"
    )


@pytest.mark.dataset
def test_episode_safe_split_requires_two_episodes(tmp_path: Path):
    np.savez_compressed(
        tmp_path / "only_episode.npz",
        obs=np.zeros((20, 64, 64, 3), dtype=np.uint8),
        action=np.zeros((20, 3), dtype=np.float32),
        reward=np.zeros((20,), dtype=np.float32),
        done=np.zeros((20,), dtype=bool),
    )

    with pytest.raises(ValueError, match="at least two rollout files"):
        episode_safe_train_val_split(tmp_path)
