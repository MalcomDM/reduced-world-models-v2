import numpy as np
from pathlib import Path
import pytest

from torch.utils.data import DataLoader

from rwm.data.rollout_dataset import RolloutDataset
from rwm.trainers.deterministic.world_model_trainer import WorldModelTrainer


# Helper to generate a tiny synthetic rollout
def make_dummy_rollout(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    T = 20
    obs = np.zeros((T, 64, 64, 3), dtype=np.uint8)
    action = np.zeros((T, 3), dtype=np.float32)
    # simple ramp reward
    reward = np.linspace(0, 1, T, dtype=np.float32)
    done = np.zeros(T, dtype=bool)
    np.savez_compressed(path, obs=obs, action=action, reward=reward, done=done)


@pytest.mark.integration
def test_world_model_trainer_end_to_end(tmp_path: Path):    
	# 1) Create two dummy rollouts
	rollout_dir = tmp_path / "data"
	for i in range(2):
		make_dummy_rollout(rollout_dir / f"rollout_{i}.npz")

	# 2) Build DataLoader from that folder
	ds 		= RolloutDataset( rollout_dir, sequence_len=16, image_size=64, include_done=False )
	loader  = DataLoader( ds, batch_size=4, shuffle=True, drop_last=True )

	# 3) Instantiate & run the trainer for 2 epochs
	out_dir = tmp_path / "runs"
	trainer = WorldModelTrainer(
		loader=loader,
		out_dir=out_dir,
		sequence_len=16,
		epochs=2,
		batch_size=4,
		lr=1e-3,
		alpha=1.0,
		beta=0.1,
	)
	best_model_path = trainer.fit()

	# 4) Check that the checkpoint and metrics file exist
	assert best_model_path.exists(), "Expected best_world_model.pt to be written"
	metrics_path = out_dir / "metrics.csv"
	assert metrics_path.exists(), "Expected metrics.csv to be written"

	# 5) Sanity-check the metrics file has at least a header + two rows (for 2 epochs)
	lines = metrics_path.read_text().strip().splitlines()
	assert len(lines) >= 3, f"metrics.csv should have >=3 lines (header+2), got {len(lines)}"
