#!/usr/bin/env python3
"""Factual masked-dynamics evaluation for Stage 2.5D.

Evaluates the K=8 reward anchors under controlled observation masking:
warmup steps are visible; from warmup onward the spatial representation is
zeroed for each mask horizon.  Three action-history variants isolate the
contribution of previous-action context.

Usage::

    python scripts/evaluation/evaluate_masked_dynamics.py \\
        --checkpoint runs/component_refinement/causal_transformer/05_k_ablation/learned_k8_seed42/checkpoint_best.pt \\
        --data-split-seed 42 --cache-dir data/cache/rollout_frames_v1 \\
        --out runs/component_refinement/causal_transformer/07_masked_factual_dynamics/seed42.json
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
_config_value = _eval_ckpt._config_value


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


def _temporal_mask_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    training = (
        config.get("training")
        if isinstance(config, dict)
        else getattr(config, "training", None)
    )
    if training is None:
        return default
    temporal_mask = (
        training.get("temporal_mask")
        if isinstance(training, dict)
        else getattr(training, "temporal_mask", None)
    )
    if temporal_mask is None:
        return default
    if isinstance(temporal_mask, dict):
        return temporal_mask.get(name, default)
    return getattr(temporal_mask, name, default)


def _resolve_recurrent_layout(config: Any) -> tuple[str, bool, int]:
    temporal = _config_value(config, "temporal", None)
    backend = _config_value(temporal, "backend", "causal_transformer")
    recurrent_context = backend == "minimal_sru"
    burn_in_steps = (
        int(_config_value(temporal, "sru_burn_in_steps", 0))
        if recurrent_context
        else 0
    )
    return backend, recurrent_context, burn_in_steps


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
    parser.add_argument(
        "--mask-horizons",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 12],
        help="Target-relative blind horizons (default: 1 2 4 8 12).",
    )
    parser.add_argument(
        "--observation-dropout-execution",
        choices=("post_perception", "pre_perception_skip"),
        default=None,
        help="Runtime execution override; defaults to the checkpoint policy.",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        parser.error(f"Refusing to overwrite existing output: {out_path}")

    # Load checkpoint and resolve config
    ckpt = load_checkpoint(args.checkpoint)
    config = ckpt.get("config")
    saved_policy = _perception_value(config, "tokenizer_eval_mode", "sample")
    saved_execution = _temporal_mask_value(
        config, "observation_dropout_execution", "post_perception",
    )
    runtime_execution = args.observation_dropout_execution or saved_execution
    backend, recurrent_context, burn_in_steps = _resolve_recurrent_layout(config)
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
    print(
        "Dropout execution: "
        f"saved={saved_execution}, runtime={runtime_execution}"
    )

    # Build data loaders (same split as original training)
    train_loader, val_loader, data_info = build_protocol_loaders(
        args.data_root, sequence_len=args.sequence_len, batch_size=args.batch_size,
        max_val_windows=args.max_val_windows, data_split_seed=data_split_seed,
        cache_dir=args.cache_dir,
        recurrent_context=recurrent_context,
        burn_in_steps=burn_in_steps,
    )
    train_mean = reward_mean(train_loader)
    print(f"Train reward mean: {train_mean:.6f}, val windows: {data_info['val_windows']}")

    # Run masked evaluation
    evaluator = MaskedFactualEvaluator(
        model,
        device,
        train_reward_mean=train_mean,
        tokenizer_eval_mode="mean",
        observation_dropout_execution=runtime_execution,
    )
    start = time.time()
    summary = evaluator.evaluate(
        val_loader,
        warmup=args.warmup,
        mask_horizons=tuple(args.mask_horizons),
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
        "temporal_backend": backend,
        "max_val_windows": args.max_val_windows,
        "warmup": args.warmup,
        "mask_horizons": args.mask_horizons,
        "train_reward_mean": train_mean,
        "tokenizer_policy_saved": saved_policy,
        "tokenizer_policy_runtime": runtime_policy,
        "observation_dropout_execution_saved": saved_execution,
        "observation_dropout_execution_runtime": runtime_execution,
        "mask_anchor": "loss_mask",
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
