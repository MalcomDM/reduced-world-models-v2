#!/usr/bin/env python3
"""Thin CLI for Stage 5.x frozen-world-model imagined Actor-Critic training.

Usage::

    # Fixed-horizon training
    python scripts/train_imagined_actor_critic.py --checkpoint <path> --horizon 4

    # Curriculum training (sample H from {1, 2, 4} per batch)
    python scripts/train_imagined_actor_critic.py --checkpoint <path> \\
        --horizons 1 2 4 --seed 42

    # Smoke test
    python scripts/train_imagined_actor_critic.py --checkpoint <path> --smoke

The checkpoint must be a Stage-2.5D.1 masked-trained anchor
(e.g. ``runs/component_refinement/causal_transformer/08_masked_reward_anchor/seed42/checkpoint_best.pt``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rwm.config.config import ACTION_DIM
from rwm.config.experiment_config import ExperimentConfig
from rwm.data.rollout_dataset import RolloutDataset
from rwm.trainers.imagined_actor_critic import (
    ImaginedACTrainer,
    ImaginedACTrainingConfig,
)
from rwm.utils.checkpointing import load_checkpoint, model_from_checkpoint
from rwm.utils.run_directory import create_run_directory


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _reserve_probe_batch(loader, device):
    """Grab one batch from the loader for fixed-probe evaluation."""
    batch = next(iter(loader))
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen-world-model imagined Actor-Critic training",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the frozen world-model .pt checkpoint",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run a short smoke test (B=2, H=4, max_batches=10)",
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=None,
        help="Entropy coefficient override (default: 0.001)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="",
        help="Override the rollout data directory",
    )
    parser.add_argument(
        "--out", type=str, default="",
        help="Output directory (default: runs/imagined_actor_critic/<name>)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="Override batch size",
    )
    parser.add_argument(
        "--horizon", type=int, default=0,
        help="Fixed imagination horizon (1-12; overrides curriculum)",
    )
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=None,
        help="Curriculum of horizons to sample from per batch (e.g. 1 2 4)",
    )
    parser.add_argument(
        "--max-batches", type=int, default=0,
        help="Override max batches",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="",
        help="Explicit frame-cache directory; omitted means uncached loading.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    ckpt_hash = _file_hash(ckpt_path)
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"  SHA256 (first 16): {ckpt_hash}")

    ckpt = load_checkpoint(ckpt_path)
    model = model_from_checkpoint(ckpt, action_dim=ACTION_DIM,
                                   tokenizer_eval_mode_override="mean")
    model.eval()
    print(f"  Model loaded (reward_head_kind={model._reward_head_kind})")

    # ------------------------------------------------------------------
    # Training config
    # ------------------------------------------------------------------
    train_cfg = ImaginedACTrainingConfig()

    if args.smoke:
        train_cfg.batch_size = 2
        train_cfg.imagination_horizon = 4
        train_cfg.max_batches = 10
        train_cfg.log_every = 1

    if args.horizons is not None:
        train_cfg.horizons = args.horizons
    if args.horizon > 0:
        train_cfg.imagination_horizon = args.horizon
        train_cfg.horizons = None  # explicit override disables curriculum
    if args.max_batches > 0:
        train_cfg.max_batches = args.max_batches
    if args.batch_size > 0:
        train_cfg.batch_size = args.batch_size
    if args.entropy_coef is not None:
        train_cfg.entropy_coef = args.entropy_coef

    train_cfg.validate()

    pool_str = (str(train_cfg.active_horizon_pool) if train_cfg.horizons
                else str(train_cfg.imagination_horizon))
    print(f"  Config: H={pool_str}, B={train_cfg.batch_size}, "
          f"steps={train_cfg.max_batches}, seed={args.seed}")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    assert ckpt["config"] is not None, "Checkpoint must have a config"
    exp_config: ExperimentConfig = ckpt["config"]
    data_dir = args.data_dir or exp_config.data.dataset_dir

    cache_dir = args.cache_dir
    if cache_dir and not Path(cache_dir).is_dir():
        parser.error(f"--cache-dir does not exist or is not a directory: {cache_dir}")
    seq_len = exp_config.data.sequence_len

    dataset = RolloutDataset(
        root_dir=Path(data_dir),
        sequence_len=seq_len,
        image_size=exp_config.data.image_size,
        cache_dir=cache_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )
    print(f"  Dataset: {len(dataset)} windows from {data_dir}")
    if cache_dir:
        print(f"  Cache: {cache_dir}")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    if args.out:
        output_dir = Path(args.out)
        run_id = output_dir.name
    else:
        run_id = time.strftime("run_%Y%m%d_%H%M%S")
        output_dir = Path("runs/imagined_actor_critic") / run_id
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")
    out_dir = create_run_directory(
        "imagined_actor_critic", exp_config, run_id=run_id, output_dir=output_dir,
    )
    print(f"  Output: {out_dir}")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ------------------------------------------------------------------
    # Probe batch (reserved before trainer consumes the loader)
    # ------------------------------------------------------------------
    probe_batch = _reserve_probe_batch(loader, device)
    print(f"  Probe batch reserved (B={probe_batch['obs'].shape[0]})")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = ImaginedACTrainer(
        model=model,
        train_loader=loader,
        train_cfg=train_cfg,
        device=device,
        out_dir=out_dir,
        seed=args.seed,
        probe_batch=probe_batch,
    )
    trainer.set_anchor_info(str(ckpt_path.resolve()), ckpt_hash)

    # ------------------------------------------------------------------
    # Pre-training fixed probe
    # ------------------------------------------------------------------
    print("\n--- Pre-training fixed probe ---")
    pre_probe = trainer.evaluate_fixed_probe(horizons=(1, 2, 4))
    print(json.dumps(pre_probe, indent=2))
    with open(out_dir / "fixed_probe_pre.json", "w") as f:
        json.dump(pre_probe, f, indent=2, sort_keys=True)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("\nStarting training...")
    trainer.train()

    # ------------------------------------------------------------------
    # Post-training fixed probe
    # ------------------------------------------------------------------
    print("\n--- Post-training fixed probe ---")
    post_probe = trainer.evaluate_fixed_probe(horizons=(1, 2, 4))
    print(json.dumps(post_probe, indent=2))
    with open(out_dir / "fixed_probe_post.json", "w") as f:
        json.dump(post_probe, f, indent=2, sort_keys=True)

    print(f"\nDone. Output in {out_dir}")


if __name__ == "__main__":
    main()
