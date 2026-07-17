import pytest
import numpy as np
from pathlib import Path
import importlib.util

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


import torch


def _build_frame_cache(data_root: Path, cache_dir: Path, **kwargs):
    """Load the standalone cache-builder script without assuming scripts is a package."""
    script_path = Path(__file__).parents[2] / "scripts" / "build_frame_cache.py"
    spec = importlib.util.spec_from_file_location("build_frame_cache_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_cache(data_root=data_root, cache_dir=cache_dir, **kwargs)

# ---------------------------------------------------------------------------
# Frame-cache tests
# ---------------------------------------------------------------------------

@pytest.mark.dataset
def test_cache_parity(tmp_path: Path):
    """Cached and uncached dataset samples must match exactly."""
    src = tmp_path / "data"
    src.mkdir()
    rng = np.random.RandomState(0)
    obs = rng.randint(0, 256, (30, 64, 64, 3), dtype=np.uint8)
    act = rng.uniform(-1, 1, (30, 3)).astype(np.float32)
    rew = rng.randn(30).astype(np.float32)
    don = np.zeros(30, dtype=bool)
    np.savez_compressed(src / "ep.npz", obs=obs, action=act, reward=rew, done=don)

    cache_dir = tmp_path / "cache"
    _build_frame_cache(data_root=src, cache_dir=cache_dir, dry_run=False)

    ds_u = RolloutDataset.from_file_list([src / "ep.npz"], sequence_len=8)
    ds_c = RolloutDataset.from_file_list([src / "ep.npz"], sequence_len=8, cache_dir=cache_dir)

    assert len(ds_u) == len(ds_c)
    for i in range(min(len(ds_u), 5)):
        u = ds_u[i]; c = ds_c[i]
        torch.testing.assert_close(u["obs"], c["obs"], msg=f"i={i} obs")
        torch.testing.assert_close(u["action"], c["action"], msg=f"i={i} action")

@pytest.mark.dataset
def test_cache_missing_entry_fails(tmp_path: Path):
    """Accessing a source not in the cache manifest must raise."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Write a manifest with data_root but no matching entry
    import json
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump({
            "schema_version": 1, "image_size": 64,
            "transform_spec": "ToTensor+Resize",
            "data_root": str(tmp_path.resolve()),
            "file_map": {},
        }, f)
    src = tmp_path / "unknown.npz"
    np.savez_compressed(src, obs=np.zeros((20, 64, 64, 3), dtype=np.uint8),
                        action=np.zeros((20, 3)), reward=np.zeros(20), done=np.zeros(20, dtype=bool))
    ds = RolloutDataset.from_file_list([src], sequence_len=8, cache_dir=cache_dir)
    with pytest.raises(ValueError, match="not found in cache"):
        _ = ds[0]

@pytest.mark.dataset
def test_cache_source_modification_invalidates(tmp_path: Path):
    """Changing the source file after caching must produce a key mismatch error."""
    src = tmp_path / "data"
    src.mkdir()
    rng = np.random.RandomState(0)
    np.savez_compressed(src / "ep.npz",
        obs=rng.randint(0, 256, (20, 64, 64, 3), dtype=np.uint8),
        action=np.zeros((20, 3), dtype=np.float32),
        reward=np.zeros(20, dtype=np.float32),
        done=np.zeros(20, dtype=bool))
    cache_dir = tmp_path / "cache"
    _build_frame_cache(data_root=src, cache_dir=cache_dir, dry_run=False)

    # Modify source
    rng2 = np.random.RandomState(99)
    np.savez_compressed(src / "ep.npz",
        obs=rng2.randint(0, 256, (20, 64, 64, 3), dtype=np.uint8),
        action=np.zeros((20, 3), dtype=np.float32),
        reward=np.zeros(20, dtype=np.float32),
        done=np.zeros(20, dtype=bool))
    with pytest.raises(ValueError, match="key mismatch|changed"):
        ds = RolloutDataset.from_file_list([src / "ep.npz"], sequence_len=8, cache_dir=cache_dir)
        _ = ds[0]

@pytest.mark.dataset
def test_cache_image_size_mismatch_fails_at_init(tmp_path: Path):
    """Requesting image_size=32 with a 64-pixel cache must fail at Dataset init."""
    src = tmp_path / "data"
    src.mkdir()
    np.savez_compressed(src / "ep.npz",
        obs=np.zeros((20, 64, 64, 3), dtype=np.uint8),
        action=np.zeros((20, 3)), reward=np.zeros(20), done=np.zeros(20, dtype=bool))
    cache_dir = tmp_path / "cache"
    _build_frame_cache(data_root=src, cache_dir=cache_dir, image_size=64)
    # The dataset constructor calls load_manifest which validates image_size.
    with pytest.raises(ValueError, match="image size"):
        RolloutDataset.from_file_list(
            [src / "ep.npz"], sequence_len=8, image_size=32, cache_dir=cache_dir,
        )

@pytest.mark.dataset
def test_custom_transform_with_cache_fails(tmp_path: Path):
    """A custom Dataset transform must be rejected when cache_dir is provided."""
    src = tmp_path / "data"
    src.mkdir()
    np.savez_compressed(src / "ep.npz",
        obs=np.zeros((20, 64, 64, 3), dtype=np.uint8),
        action=np.zeros((20, 3)), reward=np.zeros(20), done=np.zeros(20, dtype=bool))
    cache_dir = tmp_path / "cache"
    _build_frame_cache(data_root=src, cache_dir=cache_dir)
    from torchvision import transforms as T
    custom = T.Compose([T.ToTensor(), T.Resize((64, 64), antialias=True)])
    with pytest.raises(ValueError, match="custom.*transform"):
        RolloutDataset(
            file_list=[src / "ep.npz"], sequence_len=8,
            cache_dir=cache_dir, transform=custom,
        )


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------

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
