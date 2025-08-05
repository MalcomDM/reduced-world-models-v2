import pytest
import numpy as np
from pathlib import Path

from rwm.data.rollout_dataset import RolloutDataset


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