import psutil
from tqdm import trange
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader

from rwm.types import RolloutSample
from rwm.config.config import ERROR_THRESHOLD
from rwm.data.rollout_dataset import RolloutDataset
from rwm.trainers.deterministic.world_model_trainer import WorldModelTrainer


def train_world_model_loop(
    rollout_dirs: list[Path],
    out_dir: Path,
    sequence_len: int = 16,
    batch_size: int = 32,
    max_epochs: int = 50,
    image_size: int = 64,
    alpha: float = 1.0,
    beta: float = 1.0,
    error_threshold: float = ERROR_THRESHOLD,
	warmup_steps: int=20,
) -> Path:
	"""
	1) Builds a DataLoader from all scenario folders in `rollout_dirs`.
	2) Instantiates WorldModelTrainer with that loader.
	3) Runs epoch-by-epoch train → eval → log, stopping when mae_cum < error_threshold.
	"""
	# — build datasets & loader (separation of concerns) —
	datasets: List[Dataset[RolloutSample]] = [
		RolloutDataset(d, sequence_len=sequence_len, image_size=image_size)
		for d in rollout_dirs
	]
	full_dataset: Dataset[RolloutSample] = ConcatDataset(datasets)
	loader: DataLoader[RolloutSample] = DataLoader(
		full_dataset,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True,
		num_workers=4,
		pin_memory=True, 
	)

	# Use WorldModelTrainer
	trainer = WorldModelTrainer(
		loader=loader,
		out_dir=out_dir,
		sequence_len=sequence_len,
		batch_size=batch_size,
		epochs=max_epochs,
        lr=3e-4,
		alpha=alpha,
		beta=beta,
		warmup_steps=warmup_steps
	)

	for epoch in trange(1, max_epochs + 1, desc="Epochs"):
		
		if torch.cuda.is_available():						# — reset GPU peak stats —
			torch.cuda.reset_peak_memory_stats()

		# — train one epoch —
		train_loss, elapsed = trainer.train_one_epoch()

		# — now capture resource usage —
		gpu_peak = (
			torch.cuda.max_memory_allocated() / 1e9
			if torch.cuda.is_available() else 0.0
		)
		proc = psutil.Process()
		cpu_rss = proc.memory_info().rss / 1e9

		if epoch % 5 == 0 or epoch == 1 or epoch == max_epochs or train_loss < error_threshold:
			eval_metrics = trainer.evaluate()
		else:
			eval_metrics = {"mae_cum": float("inf"), "mae_step": float("inf")}

		# merge into one row
		row: Dict[str, float] = {
			"epoch": epoch,
			"train_loss": train_loss,
			"mae_cum": eval_metrics["mae_cum"],
			"mae_step": eval_metrics["mae_step"],
			"time": elapsed,
			"gpu_mem_peak_gb": gpu_peak,
			"cpu_mem": cpu_rss
		}
		trainer.log_and_checkpoint(epoch, row)

		if train_loss < error_threshold:
			break

	return out_dir / "best_world_model.pt"

