#!/usr/bin/env python3
"""Thin CLI for Stage 5.0 frozen-world-model imagined Actor-Critic training.

Usage::

    python scripts/train_imagined_actor_critic.py --checkpoint <path> --smoke

The checkpoint must be a Stage-2.5D.1 masked-trained anchor
(e.g. ``runs/component_refinement/08_masked_reward_anchor/seed42/checkpoint_best.pt``).
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen-world-model imagined Actor-Critic training (Stage 5.0)",
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
        "--data-dir", type=str, default="",
        help="Override the rollout data directory",
    )
    parser.add_argument(
        "--out", type=str, default="",
        help="Output directory (default: runs/imagined_ac/<timestamp>)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="Override batch size",
    )
    parser.add_argument(
        "--horizon", type=int, default=0,
        help="Override imagination horizon (1-12; default 4)",
    )
    parser.add_argument(
        "--max-batches", type=int, default=0,
        help="Override max batches",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="",
        help="Explicit frame-cache directory; omitted means uncached loading.",
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

    if args.horizon > 0:
        train_cfg.imagination_horizon = args.horizon
    if args.max_batches > 0:
        train_cfg.max_batches = args.max_batches
    if args.batch_size > 0:
        train_cfg.batch_size = args.batch_size

    train_cfg.validate()
    print(f"  Config: H={train_cfg.imagination_horizon}, "
          f"B={train_cfg.batch_size}, steps={train_cfg.max_batches}")

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
    # Trainer
    # ------------------------------------------------------------------
    trainer = ImaginedACTrainer(
        model=model,
        train_loader=loader,
        train_cfg=train_cfg,
        device=device,
        out_dir=out_dir,
    )
    trainer.set_anchor_info(str(ckpt_path.resolve()), ckpt_hash)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    print("\nStarting training...")
    trainer.train()

    print(f"\nDone. Output in {out_dir}")


if __name__ == "__main__":
    main()
