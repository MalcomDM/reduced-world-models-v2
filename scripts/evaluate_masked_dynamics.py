#!/usr/bin/env python3
"""Factual masked-dynamics evaluation for Stage 2.5D.

Evaluates the K=8 reward anchors under controlled observation masking:
warmup steps are visible; from warmup onward the spatial representation is
zeroed for each mask horizon.  Three action-history variants isolate the
contribution of previous-action context.

Usage::

    python scripts/evaluate_masked_dynamics.py \\
        --checkpoint runs/component_refinement/05_k_ablation/learned_k8_seed42/checkpoint_best.pt \\
        --data-split-seed 42 --cache-dir data/cache/rollout_frames_v1 \\
        --out runs/component_refinement/07_masked_factual_dynamics/seed42.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from rwm.config.config import ACTION_DIM
from rwm.evaluation.masked_factual_evaluator import MaskedFactualEvaluator
from rwm.utils.checkpointing import load_checkpoint, model_from_checkpoint

# Reuse protocol loaders from the checkpoint evaluator.
_SCRIPT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_eval_ckpt", _SCRIPT_DIR / "evaluate_checkpoint.py")
_eval_ckpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_ckpt)
build_protocol_loaders = _eval_ckpt.build_protocol_loaders
reward_mean = _eval_ckpt.reward_mean


DATA_ROOT = Path("data/rollouts/rwm_deterministic/scenario_0")


def _perception_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    pc = config.get("perception") if isinstance(config, dict) else getattr(config, "perception", None)
    if pc is None:
        return default
    if isinstance(pc, dict):
        return pc.get(name, default)
    return getattr(pc, name, default)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Masked factual dynamics evaluation (Stage 2.5D).",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Exact JSON output path (refuses overwrite).")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--data-split-seed", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-len", type=int, default=16)
    parser.add_argument("--max-val-windows", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        parser.error(f"Refusing to overwrite existing output: {out_path}")

    # Load checkpoint and resolve config
    ckpt = load_checkpoint(args.checkpoint)
    config = ckpt.get("config")
    saved_policy = _perception_value(config, "tokenizer_eval_mode", "sample")
    data_split_seed = args.data_split_seed if args.data_split_seed is not None else int(
        config.get("seed", 42) if isinstance(config, dict) else getattr(config, "seed", 42)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model with forced mean policy
    model = model_from_checkpoint(
        ckpt, action_dim=ACTION_DIM,
        tokenizer_eval_mode_override="mean",
    ).to(device)
    model.eval()
    runtime_policy = model.tokenizer.eval_mode
    print(f"Tokenizer policy: saved={saved_policy}, runtime={runtime_policy}")

    # Build data loaders (same split as original training)
    train_loader, val_loader, data_info = build_protocol_loaders(
        args.data_root, sequence_len=args.sequence_len, batch_size=args.batch_size,
        max_val_windows=args.max_val_windows, data_split_seed=data_split_seed,
        cache_dir=args.cache_dir,
    )
    train_mean = reward_mean(train_loader)
    print(f"Train reward mean: {train_mean:.6f}, val windows: {data_info['val_windows']}")

    # Run masked evaluation
    evaluator = MaskedFactualEvaluator(model, device, train_reward_mean=train_mean, tokenizer_eval_mode="mean")
    start = time.time()
    summary = evaluator.evaluate(
        val_loader,
        warmup=args.warmup,
        mask_horizons=(1, 2, 4, 8, 16),
        action_variants=("correct", "zero", "shifted"),
    )
    elapsed = time.time() - start

    # Build output
    output = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data_root": str(args.data_root.resolve()),
        "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else "",
        "data_split_seed": data_split_seed,
        "batch_size": args.batch_size,
        "sequence_len": args.sequence_len,
        "max_val_windows": args.max_val_windows,
        "warmup": args.warmup,
        "train_reward_mean": train_mean,
        "tokenizer_policy_saved": saved_policy,
        "tokenizer_policy_runtime": runtime_policy,
        "elapsed_s": elapsed,
        **data_info,
        **summary,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
