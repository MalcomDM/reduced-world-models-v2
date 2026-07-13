"""High-level world-model training loop with episode-safe train/val split."""

import psutil
from tqdm import trange
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import DataLoader

from rwm.data.rollout_dataset import (
    build_train_val_datasets,
    _collect_npz_files,
)
from rwm.trainers.deterministic.world_model_trainer import WorldModelTrainer
from rwm.config.experiment_config import ExperimentConfig
from rwm.utils.dataset_manifest import (
    build_dataset_manifest,
    save_manifest,
    validate_manifest,
)


def train_world_model_loop(
    rollout_dirs: list[Path],
    out_dir: Path,
    sequence_len: int = 16,
    batch_size: int = 32,
    max_epochs: int = 50,
    image_size: int = 64,
    beta: float = 1.0,
    warmup_steps: int = 20,
    val_ratio: float = 0.2,
    config: Optional[ExperimentConfig] = None,
) -> Path:
    """
    Builds train/validation datasets using episode-safe file-level split,
    then runs epoch-by-epoch training with held-out validation.
    """
    # Merge all scenario directories into one root for the dataset.
    # create a synthetic root that includes all files from all scenarios
    all_files: List[Path] = []
    for d in rollout_dirs:
        all_files.extend(_collect_npz_files(d))

    if len(all_files) < 2:
        raise ValueError(
            f"Need at least 2 rollout files for train/val split; "
            f"found {len(all_files)} across {len(rollout_dirs)} scenario dirs."
        )

    rng = __import__("numpy").random.RandomState(42)
    rng.shuffle(all_files)
    n_val = max(1, int(len(all_files) * val_ratio))
    train_files, val_files = all_files[n_val:], all_files[:n_val]

    from rwm.data.rollout_dataset import RolloutDataset
    train_ds = RolloutDataset.from_file_list(
        train_files, sequence_len=sequence_len, image_size=image_size,
    )
    val_ds = RolloutDataset.from_file_list(
        val_files, sequence_len=sequence_len, image_size=image_size,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=2, pin_memory=True,
    )

    # Dataset manifest
    manifest_ref: Optional[str] = None
    if config is not None:
        try:
            manifest = build_dataset_manifest(
                data_root=rollout_dirs[0].parent if len(rollout_dirs) == 1
                else rollout_dirs[0],
                sequence_len=sequence_len,
                image_size=image_size,
                val_ratio=val_ratio,
            )
            issues = validate_manifest(manifest)
            if issues:
                print(f"Warning: dataset manifest issues: {issues}")
            manifest_path = out_dir / "dataset_manifest.json"
            save_manifest(manifest, manifest_path)
            manifest_ref = manifest_path.name
        except Exception as exc:
            print(f"Warning: could not build dataset manifest: {exc}")

    trainer = WorldModelTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        out_dir=out_dir,
        sequence_len=sequence_len,
        epochs=max_epochs,
        batch_size=batch_size,
        lr=3e-4,
        beta=beta,
        warmup_steps=warmup_steps,
        config=config,
        dataset_manifest_ref=manifest_ref,
    )

    for epoch in trange(1, max_epochs + 1, desc="Epochs"):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_loss, elapsed = trainer.train_one_epoch()

        gpu_peak = (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available() else 0.0
        )
        proc = psutil.Process()
        cpu_rss = proc.memory_info().rss / 1e9

        val_metrics = trainer.evaluate()

        row: Dict[str, float] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mse": val_metrics["val_mse"],
            "time": elapsed,
            "gpu_mem_peak_gb": gpu_peak,
            "cpu_mem": cpu_rss,
        }
        trainer.log_and_checkpoint(epoch, row)

    return out_dir / "best_world_model.pt"
