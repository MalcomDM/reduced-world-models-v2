#!/usr/bin/env python3
"""Profile real-data training pipeline: loader, transfer, compute, validation.

Measures one epoch with CUDA synchronization, reporting:
  - dataset/loader wait time
  - host-to-device transfer time
  - model forward+backward+optimizer time
  - validation time
  - total epoch time, windows/sec, frames/sec, peak GPU memory

Usage:
    python scripts/profiling/profile_loader.py --num-workers 2 --pin-memory 1
    python scripts/profiling/profile_loader.py --num-workers 0
    python scripts/profiling/profile_loader.py --num-workers 4 --pin-memory 1 --persistent-workers
"""

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np

from rwm.data.rollout_dataset import RolloutDataset, _collect_npz_files
from rwm.trainers.deterministic.world_model_trainer import WorldModelTrainer
from rwm.config.experiment_config import ExperimentConfig, DataConfig, TrainingConfig
from rwm.utils.seeding import set_seed
from typing import Optional
from torch.utils.data import DataLoader

DATA_ROOT = Path("data/rollouts/rwm_deterministic/scenario_0")


def profile_epoch(
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    batch_size: int = 8,
    sequence_len: int = 16,
    seed: int = 42,
    val_ratio: float = 0.2,
    max_val_windows: int = 256,
    out_dir: Path = Path("/tmp/loader_profile"),
    cache_dir: Optional[Path] = None,
) -> dict:
    set_seed(seed)

    # Build datasets
    all_files = _collect_npz_files(DATA_ROOT)
    rng = np.random.RandomState(seed)
    rng.shuffle(all_files)
    n_val = max(1, int(len(all_files) * val_ratio))
    train_files = all_files[n_val:]
    val_files = all_files[:n_val]

    # cache_dir is passed explicitly from args; never auto-detect.
    train_ds = RolloutDataset.from_file_list(
        train_files, sequence_len=sequence_len, cache_dir=cache_dir,
    )

    # Val subset
    val_ds = RolloutDataset.from_file_list(
        val_files, sequence_len=sequence_len, cache_dir=cache_dir,
    )
    rng2 = np.random.RandomState(seed)
    n_v = min(max_val_windows, len(val_ds))
    val_indices = rng2.choice(len(val_ds), size=n_v, replace=False).tolist()
    from torch.utils.data import Subset
    val_subset = Subset(val_ds, val_indices)

    # Loader
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=0, pin_memory=False,
    )

    # Trainer
    cfg = ExperimentConfig(
        experiment_name="loader_profile",
        run_id=f"nw{num_workers}_pin{pin_memory}",
        seed=seed,
        training=TrainingConfig(batch_size=batch_size, beta=0.1),
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = WorldModelTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        out_dir=out_dir,
        sequence_len=sequence_len,
        epochs=1, batch_size=batch_size,
        lr=3e-4, beta=0.1,
        config=cfg,
    )

    # Warmup loader
    print("Warming up loader...")
    warmup_batch = next(iter(train_loader))
    _ = warmup_batch["obs"].to(trainer.device)
    del warmup_batch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Profile one epoch
    model = trainer.model.train()
    optimizer = trainer.optimizer

    total_loader_wait = 0.0
    total_transfer = 0.0
    total_compute = 0.0
    n_batches = 0

    t_epoch_start = time.perf_counter()

    for batch in train_loader:
        # Loader wait: time to get batch from loader
        t_loader = time.perf_counter()

        # Host-to-device transfer
        obs = batch["obs"].to(trainer.device, non_blocking=True)
        act = batch["action"].to(trainer.device, non_blocking=True)
        rew = batch["reward"].to(trainer.device, non_blocking=True)
        predecessor_action = batch.get("predecessor_action")
        device_batch = {
            "obs": obs,
            "action": act,
            "reward": rew,
            "done": batch["done"].to(trainer.device, non_blocking=True),
        }
        if predecessor_action is not None:
            device_batch["predecessor_action"] = predecessor_action.to(
                trainer.device, non_blocking=True,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_transfer = time.perf_counter()

        # Compute: forward + backward + step
        optimizer.zero_grad()
        # Pass the transferred batch so compute timing excludes host-to-device IO.
        loss_total, loss_mse, loss_kl = trainer._compute_batch_loss(device_batch)
        loss_total.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_compute = time.perf_counter()

        total_loader_wait += t_loader - t_epoch_start if n_batches == 0 else t_loader - t_compute_prev
        total_transfer += t_transfer - t_loader
        total_compute += t_compute - t_transfer
        n_batches += 1
        t_compute_prev = t_compute

    t_epoch_end = time.perf_counter()
    epoch_time = t_epoch_end - t_epoch_start

    # Validation
    t_val_start = time.perf_counter()
    val_metrics = trainer.evaluate()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_val_end = time.perf_counter()
    val_time = t_val_end - t_val_start

    peak_gpu = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    n_windows = n_batches * batch_size
    results = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "batch_size": batch_size,
        "sequence_len": sequence_len,
        "n_train_batches": n_batches,
        "n_train_windows": n_windows,
        "epoch_time_s": round(epoch_time, 2),
        "loader_wait_s": round(total_loader_wait, 2),
        "transfer_s": round(total_transfer, 2),
        "compute_s": round(total_compute, 2),
        "val_time_s": round(val_time, 2),
        "loader_pct": round(100 * total_loader_wait / max(1e-6, epoch_time), 1),
        "compute_pct": round(100 * total_compute / max(1e-6, epoch_time), 1),
        "windows_per_sec": round(n_windows / max(1e-6, epoch_time), 1),
        "frames_per_sec": round(n_windows * sequence_len / max(1e-6, epoch_time), 1),
        "peak_gpu_gb": round(peak_gpu, 4),
        "val_mse": val_metrics.get("val_mse", 0),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", type=int, default=1)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-len", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Path to frame cache. Default: no cache.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Explicitly disable cache (overrides --cache-dir).")
    parser.add_argument("--out", type=Path, default=Path("runs/component_refinement/causal_transformer/benchmarks/loader_profile.json"))
    args = parser.parse_args()

    cache_dir = None if args.no_cache else args.cache_dir

    results = profile_epoch(
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        persistent_workers=args.persistent_workers,
        batch_size=args.batch_size,
        sequence_len=args.sequence_len,
        cache_dir=cache_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
