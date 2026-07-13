#!/usr/bin/env python3
"""Stage 2 reward-model validation experiment.

Usage:

    # Bounded held-out validation (beta=0, reward-only):
    python scripts/validate_stage2.py --beta 0 --epochs 15 --out runs/expA \\
        --max-val-windows 128

    # KL-weighted validation (same split):
    python scripts/validate_stage2.py --beta 1.0 --epochs 15 --out runs/expB \\
        --max-val-windows 128

    # Action probe on a trained checkpoint:
    python scripts/validate_stage2.py --checkpoint runs/expA/checkpoint_best.pt

    # Profile eval throughput:
    python scripts/validate_stage2.py --profile --max-val-windows 256 --out /tmp/profile

    # Short smoke:
    python scripts/validate_stage2.py --smoke --out runs/val_smoke
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from rwm.data.rollout_dataset import (
    RolloutDataset,
    _collect_npz_files,
)
from rwm.trainers.deterministic.world_model_trainer import WorldModelTrainer
from rwm.config.experiment_config import ExperimentConfig, DataConfig, TrainingConfig
from rwm.utils.run_directory import create_run_directory
from rwm.utils.dataset_manifest import build_dataset_manifest, save_manifest
from rwm.utils.seeding import set_seed
from rwm.utils.probe_set import make_default_probe, save_probe_set
from rwm.utils.checkpointing import load_checkpoint


DATA_ROOT = Path("data/rollouts/rwm_deterministic/scenario_0")


def _collect_and_split(
    root: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    max_train_files: Optional[int] = None,
    overfit_file: Optional[Path] = None,
):
    all_files = _collect_npz_files(root)
    if overfit_file is not None:
        return [overfit_file], []
    rng = np.random.RandomState(seed)
    rng.shuffle(all_files)
    n_val = max(1, int(len(all_files) * val_ratio))
    train_files = all_files[n_val:]
    val_files = all_files[:n_val]
    if max_train_files is not None:
        train_files = train_files[:max_train_files]
    return train_files, val_files


def _bounded_val_loader(val_files, sequence_len, batch_size, max_val_windows, seed):
    """Create a DataLoader limited to max_val_windows windows."""
    ds = RolloutDataset.from_file_list(val_files, sequence_len=sequence_len)
    n = min(max_val_windows, len(ds))
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), size=n, replace=False).tolist()
    subset = Subset(ds, indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=1, pin_memory=True,
    )
    return loader, n


def run(
    out_dir: Path,
    seed: int = 42,
    sequence_len: int = 16,
    batch_size: int = 8,
    max_epochs: int = 15,
    lr: float = 3e-4,
    beta: float = 0.0,
    val_ratio: float = 0.2,
    smoke: bool = False,
    max_train_files: Optional[int] = None,
    max_val_windows: Optional[int] = None,
    profile: bool = False,
) -> Dict:
    set_seed(seed)

    cfg = ExperimentConfig(
        experiment_name="stage2-val",
        run_id=f"seed{seed}",
        seed=seed,
        data=DataConfig(
            dataset_dir=str(DATA_ROOT.resolve()),
            sequence_len=sequence_len,
        ),
        training=TrainingConfig(
            batch_size=batch_size,
            max_epochs=max_epochs if not profile else 1,
            learning_rate=lr,
            beta=beta,
        ),
    )

    run_dir = create_run_directory("stage2-val", cfg, run_id=f"seed{seed}_beta{beta}")
    out_dir = run_dir
    print(f"Run directory: {out_dir}")

    probe_path = out_dir / "probes" / "default_probe.npz"
    make_default_probe(probe_path)

    train_files, val_files = _collect_and_split(
        DATA_ROOT, val_ratio=val_ratio, seed=seed,
        max_train_files=max_train_files,
    )

    train_ds = RolloutDataset.from_file_list(
        train_files, sequence_len=sequence_len,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=2, pin_memory=True,
    )

    val_loader = None
    actual_val_windows = 0
    if val_files:
        if max_val_windows is not None:
            val_loader, actual_val_windows = _bounded_val_loader(
                val_files, sequence_len, batch_size, max_val_windows, seed,
            )
        else:
            ds = RolloutDataset.from_file_list(val_files, sequence_len=sequence_len)
            actual_val_windows = len(ds)
            val_loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                drop_last=False, num_workers=1, pin_memory=True,
            )

    print(f"Train: {len(train_files)} files, {len(train_ds)} windows")
    print(f"Val:   {len(val_files)} files, {actual_val_windows} windows")

    # Manifest.
    try:
        manifest = build_dataset_manifest(
            DATA_ROOT, sequence_len=sequence_len,
            val_ratio=val_ratio, shuffle_seed=seed,
        )
        save_manifest(manifest, out_dir / "dataset_manifest.json")
    except Exception as e:
        print(f"Manifest warning: {e}")

    # Device / GPU mem start.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer = WorldModelTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        out_dir=out_dir,
        sequence_len=sequence_len,
        epochs=1 if profile else max_epochs,
        batch_size=batch_size,
        lr=lr,
        beta=beta,
        config=cfg,
        dataset_manifest_ref="dataset_manifest.json",
    )

    if profile:
        # Profile one epoch.
        start = time.time()
        t, m, k, _ = trainer.train_one_epoch()
        train_time = time.time() - start
        vstart = time.time()
        val_metrics = trainer.evaluate()
        val_time = time.time() - vstart
        gpu_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        print(f"\nProfile results:")
        print(f"  Training one epoch: {train_time:.2f}s")
        print(f"  Validation ({actual_val_windows} windows): {val_time:.2f}s")
        print(f"  Train throughput: {len(train_loader) / max(0.01, train_time):.1f} batches/s")
        print(f"  Val throughput: {len(val_loader) / max(0.01, val_time):.1f} batches/s" if val_loader else "")
        print(f"  Peak GPU memory: {gpu_peak:.2f} GB")
        return {"profile": True, "train_s": train_time, "val_s": val_time, "train_batches": len(train_loader), "val_batches": len(val_loader) if val_loader else 0}

    start = time.time()
    best_path = trainer.fit()
    elapsed = time.time() - start

    gpu_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    # Read final row from metrics.
    final_metrics = {"val_mse": trainer.best_val_metric}
    try:
        with open(out_dir / "metrics.csv") as f:
            lines = f.readlines()
        if len(lines) >= 2:
            keys = lines[0].strip().split(",")
            vals = lines[-1].strip().split(",")
            final_metrics = dict(zip(keys, vals))
    except Exception:
        pass

    print(f"\nTraining complete in {elapsed:.1f}s. Best model: {best_path}")
    print(f"Batch size: {batch_size}, sequence_len: {sequence_len}")

    results = {
        "out_dir": str(out_dir),
        "seed": seed,
        "beta": beta,
        "epochs": max_epochs,
        "train_files": len(train_files),
        "val_files": len(val_files) if val_files else 0,
        "train_windows": len(train_ds),
        "val_windows": actual_val_windows,
        "batch_size": batch_size,
        "sequence_len": sequence_len,
        "best_val_mse": trainer.best_val_metric,
        "final_val_mse": float(final_metrics.get("val_mse", trainer.best_val_metric)),
        "final_val_mae": float(final_metrics.get("val_mae", 0)),
        "final_baseline_mse": float(final_metrics.get("baseline_mse", 0)),
        # Kept as the comparison baseline for the bounded run. The training
        # reward mean is stable across epochs; a future best-checkpoint
        # evaluator can compute the exact baseline at the selected epoch.
        "baseline_mse": float(final_metrics.get("baseline_mse", 0)),
        "elapsed_s": elapsed,
        "peak_gpu_gb": gpu_peak,
        "checkpoint_best": str(out_dir / "checkpoint_best.pt"),
    }
    return results


def action_probe(checkpoint_path: Optional[Path] = None, seed: int = 42):
    """Action sensitivity probe on a trained or fresh model."""
    set_seed(seed)
    from rwm.models.rwm.model import ReducedWorldModel

    print("Loading model...")
    model = ReducedWorldModel()
    if checkpoint_path and checkpoint_path.exists():
        ckpt = load_checkpoint(checkpoint_path)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded trained checkpoint: {checkpoint_path}")
    else:
        print("Using FRESH (untrained) model")
    model.eval()

    img = torch.randn(1, 3, 64, 64)
    zero_act = torch.zeros(1, 3)
    probe_acts = [
        ("all zeros", torch.tensor([[0.0, 0.0, 0.0]])),
        ("full steer", torch.tensor([[1.0, 0.0, 0.0]])),
        ("full gas", torch.tensor([[0.0, 1.0, 0.0]])),
        ("full brake", torch.tensor([[0.0, 0.0, 1.0]])),
    ]

    with torch.no_grad():
        base = model(img=img, prev_action=zero_act, current_action=zero_act,
                     force_keep_input=True)
    belief = base.world_state

    preds = []
    with torch.no_grad():
        h = model.controller.encode(belief)
        for name, a in probe_acts:
            r = model.controller.predict_reward(h, a)
            preds.append(r.item())

    print("\nAction probe (trained model, belief-fixed):")
    for (name, _), r in zip(probe_acts, preds):
        print(f"  {name:>12s}: {r:.6f}")

    unique = len(set(round(p, 6) for p in preds))
    print(f"Unique predictions: {unique}/{len(probe_acts)}")
    return {"preds": preds, "unique": unique, "passed": unique > 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/stage2_val"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=128)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to checkpoint for action probe")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        result = action_probe(args.checkpoint, seed=args.seed)
        with open(args.out / "action_probe_results.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nDone. Results saved to {args.out / 'action_probe_results.json'}")
        return

    if args.smoke:
        args.epochs = 1
        args.max_train_files = 2
        args.max_val_windows = 32

    results = run(
        out_dir=args.out,
        seed=args.seed,
        sequence_len=args.sequence_len,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        val_ratio=args.val_ratio,
        smoke=args.smoke,
        max_train_files=args.max_train_files,
        max_val_windows=args.max_val_windows,
        profile=args.profile,
    )

    # Show model/baseline ratio.
    ratio = results.get("best_val_mse", 1) / max(1e-8, results.get("baseline_mse", 1))
    print(f"Model MSE / baseline MSE ratio: {ratio:.4f}")
    results["model_baseline_ratio"] = ratio

    with open(args.out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
