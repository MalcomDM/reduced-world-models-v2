#!/usr/bin/env python3
"""Bounded reward-prediction evaluation experiment.

Usage:

    # Bounded held-out validation (beta=0, reward-only):
    python scripts/evaluate_reward_prediction.py --beta 0 --epochs 15 --out runs/reward_baseline_a \\
        --max-val-windows 128

    # KL-weighted validation (same split):
    python scripts/evaluate_reward_prediction.py --beta 1.0 --epochs 15 --out runs/reward_kl_a \\
        --max-val-windows 128

    # Action probe on a trained checkpoint:
    python scripts/evaluate_reward_prediction.py --checkpoint runs/reward_baseline_a/checkpoint_best.pt

    # Profile eval throughput:
    python scripts/evaluate_reward_prediction.py --profile --max-val-windows 256 --out /tmp/reward_profile

    # Short smoke:
    python scripts/evaluate_reward_prediction.py --smoke --out runs/reward_smoke
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
from rwm.config.experiment_config import ExperimentConfig, DataConfig, PerceptionConfig, ControllerConfig, TrainingConfig, TemporalMaskConfig
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


def _bounded_val_loader(val_files, sequence_len, batch_size, max_val_windows, seed, cache_dir=None):
    """Create a DataLoader limited to max_val_windows windows."""
    ds = RolloutDataset.from_file_list(val_files, sequence_len=sequence_len, cache_dir=cache_dir)
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
    out_dir: Optional[Path],
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
    cache_dir: Optional[Path] = None,
    reward_head_kind: str = "linear",
    reward_head_hidden_dim: int = 32,
    selection_mode: str = "learned",
    selection_k: int = 8,
    selection_seed: int = 0,
    tokenizer_eval_mode: str = "sample",
    temporal_mask_enabled: bool = False,
    temporal_mask_warmup: int = 4,
    temporal_mask_horizons: Optional[List[int]] = None,
    temporal_mask_probability: float = 0.5,
    temporal_mask_ramp_epochs: int = 2,
) -> Dict:
    set_seed(seed)

    cfg = ExperimentConfig(
        experiment_name="reward_prediction",
        run_id=out_dir.name if out_dir is not None else "",
        seed=seed,
        data=DataConfig(
            dataset_dir=str(DATA_ROOT.resolve()),
            sequence_len=sequence_len,
            cache_dir=str(cache_dir.resolve()) if cache_dir is not None else "",
        ),
        perception=PerceptionConfig(
            k=selection_k,
            selection_mode=selection_mode,
            selection_seed=selection_seed,
            tokenizer_eval_mode=tokenizer_eval_mode,
        ),
        controller=ControllerConfig(
            reward_head_kind=reward_head_kind,
            reward_head_hidden_dim=reward_head_hidden_dim,
        ),
        training=TrainingConfig(
            batch_size=batch_size,
            max_epochs=max_epochs if not profile else 1,
            learning_rate=lr,
            beta=beta,
            temporal_mask=TemporalMaskConfig(
                enabled=temporal_mask_enabled,
                warmup_steps=temporal_mask_warmup,
                horizons=temporal_mask_horizons or [1, 2, 4, 8, 12],
                target_mask_probability=temporal_mask_probability,
                ramp_epochs=temporal_mask_ramp_epochs,
            ),
        ),
    )

    run_dir = create_run_directory(
        "reward_prediction",
        cfg,
        run_id=out_dir.name if out_dir is not None else None,
        output_dir=out_dir,
    )
    out_dir = run_dir
    print(f"Run directory: {out_dir}")

    probe_path = out_dir / "probes" / "default_probe.npz"
    make_default_probe(probe_path)

    train_files, val_files = _collect_and_split(
        DATA_ROOT, val_ratio=val_ratio, seed=seed,
        max_train_files=max_train_files,
    )

    train_ds = RolloutDataset.from_file_list(
        train_files, sequence_len=sequence_len, cache_dir=cache_dir,
    )
    nw = cfg.data.num_workers
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0,
    )

    val_loader = None
    actual_val_windows = 0
    if val_files:
        if max_val_windows is not None:
            val_loader, actual_val_windows = _bounded_val_loader(
                val_files, sequence_len, batch_size, max_val_windows, seed,
                cache_dir=cache_dir,
            )
        else:
            ds = RolloutDataset.from_file_list(
                val_files, sequence_len=sequence_len, cache_dir=cache_dir,
            )
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
        return {
            "profile": True,
            "out_dir": str(out_dir),
            "train_s": train_time,
            "val_s": val_time,
            "train_batches": len(train_loader),
            "val_batches": len(val_loader) if val_loader else 0,
        }

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


def action_probe(checkpoint_path: Optional[Path] = None, seed: int = 42,
                 tokenizer_eval_mode_override: Optional[str] = None):
    """Action sensitivity probe on a trained or fresh model."""
    set_seed(seed)
    from rwm.models.rwm.model import ReducedWorldModel
    from rwm.utils.checkpointing import model_from_checkpoint

    print("Loading model...")
    if checkpoint_path and checkpoint_path.exists():
        ckpt = load_checkpoint(checkpoint_path)
        model = model_from_checkpoint(ckpt, action_dim=3,
                                      tokenizer_eval_mode_override=tokenizer_eval_mode_override)
        saved_policy = getattr(ckpt.get("config"), "perception", None)
        if saved_policy is not None:
            saved_policy = getattr(saved_policy, "tokenizer_eval_mode", "sample")
        else:
            saved_policy = "sample"
        effective_policy = model._tokenizer_eval_mode
        print(f"Loaded trained checkpoint: {checkpoint_path}")
        print(f"  Tokenizer policy (saved): {saved_policy}")
        print(f"  Tokenizer policy (runtime): {effective_policy}")
    else:
        print("Using FRESH (untrained) model")
        model = ReducedWorldModel(action_dim=3)
        saved_policy = "sample"
        effective_policy = "sample"
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
    return {
        "preds": preds,
        "unique": unique,
        "passed": unique > 1,
        "tokenizer_policy_saved": saved_policy,
        "tokenizer_policy_runtime": effective_policy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Exact new output directory. Refuses to reuse an existing directory.",
    )
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
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Path to frame cache (default: no cache)")
    parser.add_argument("--reward-head-kind", type=str, default="linear",
                        help="Reward head architecture: linear or nonlinear")
    parser.add_argument("--reward-head-hidden-dim", type=int, default=32,
                        help="Hidden dimension for nonlinear reward head")
    parser.add_argument("--selection-mode", type=str, default="learned",
                        help="Patch selection: learned | fixed_uniform | fixed_random")
    parser.add_argument("--selection-k", type=int, default=8,
                        help="Number of selected patches")
    parser.add_argument("--selection-seed", type=int, default=0,
                        help="RNG seed for fixed_random selection")
    parser.add_argument("--tokenizer-eval-mode", type=str, default="sample",
                        choices=["sample", "mean"],
                        help="Tokenizer evaluation policy: sample (stochastic) or mean (deterministic)")
    parser.add_argument("--temporal-mask-enabled", action="store_true",
                        help="Enable temporal observation masking during training (D.1)")
    parser.add_argument("--temporal-mask-warmup", type=int, default=4,
                        help="Visible warmup steps before mask starts")
    parser.add_argument("--temporal-mask-horizons", type=int, nargs="+", default=[1, 2, 4, 8, 12],
                        help="Approved mask horizons")
    parser.add_argument("--temporal-mask-probability", type=float, default=0.5,
                        help="Target per-sample mask probability after ramp")
    parser.add_argument("--temporal-mask-ramp-epochs", type=int, default=2,
                        help="Epochs over which mask probability ramps from 0 to target")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to checkpoint for action probe")
    args = parser.parse_args()

    if args.checkpoint:
        action_probe_dir = args.out or Path("runs/reward_prediction/action_probe")
        action_probe_dir.mkdir(parents=True, exist_ok=True)
        override = args.tokenizer_eval_mode
        result = action_probe(args.checkpoint, seed=args.seed,
                              tokenizer_eval_mode_override=override)
        result["tokenizer_eval_mode_override"] = override
        with open(action_probe_dir / "action_probe_results.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nDone. Results saved to {action_probe_dir / 'action_probe_results.json'}")
        return

    if args.smoke:
        args.epochs = 1
        args.max_train_files = 2
        args.max_val_windows = 32

    try:
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
            cache_dir=args.cache_dir,
            reward_head_kind=args.reward_head_kind,
            reward_head_hidden_dim=args.reward_head_hidden_dim,
            selection_mode=args.selection_mode,
            selection_k=args.selection_k,
            selection_seed=args.selection_seed,
            tokenizer_eval_mode=args.tokenizer_eval_mode,
            temporal_mask_enabled=args.temporal_mask_enabled,
            temporal_mask_warmup=args.temporal_mask_warmup,
            temporal_mask_horizons=args.temporal_mask_horizons,
            temporal_mask_probability=args.temporal_mask_probability,
            temporal_mask_ramp_epochs=args.temporal_mask_ramp_epochs,
        )
    except FileExistsError as error:
        parser.error(
            f"Output directory already exists: {error.filename or args.out}. "
            "Choose a new --out path; experiment artifacts are never overwritten."
        )

    # Show model/baseline ratio.
    ratio = results.get("best_val_mse", 1) / max(1e-8, results.get("baseline_mse", 1))
    print(f"Model MSE / baseline MSE ratio: {ratio:.4f}")
    results["model_baseline_ratio"] = ratio

    with open(Path(results["out_dir"]) / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
