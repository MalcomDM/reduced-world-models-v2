#!/usr/bin/env python3
"""Evaluate a saved reward checkpoint on its held-out transition protocol.

This script never trains.  It reproduces the reward experiment's file split,
bounded validation windows, temporal action contract, and train-mean baseline.
``data_split_seed`` is intentionally independent from ``inference_rng_seed``:
changing posterior-sampling noise must never change the evaluated data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from rwm.config.config import ACTION_DIM
from rwm.data.rollout_dataset import RolloutDataset, _collect_npz_files
from rwm.utils.checkpointing import load_checkpoint, model_from_checkpoint
from rwm.utils.seeding import set_seed


DATA_ROOT = Path("data/rollouts/rwm_deterministic/scenario_0")


def _config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _perception_value(config: Any, name: str, default: Any) -> Any:
    return _config_value(_config_value(config, "perception", None), name, default)


def _collect_and_split(root: Path, data_split_seed: int, val_ratio: float = 0.2) -> Tuple[list[Path], list[Path]]:
    """Match ``evaluate_reward_prediction._collect_and_split`` exactly."""
    files = _collect_npz_files(root)
    if len(files) < 2:
        raise ValueError(f"Expected at least two rollout files under {root}, found {len(files)}")
    rng = np.random.RandomState(data_split_seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    return files[n_val:], files[:n_val]


def build_protocol_loaders(
    root: Path,
    *,
    sequence_len: int,
    batch_size: int,
    max_val_windows: int,
    data_split_seed: int,
    cache_dir: Optional[Path],
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Build fixed train/validation loaders without consuming inference RNG."""
    train_files, val_files = _collect_and_split(root, data_split_seed)
    train_ds = RolloutDataset.from_file_list(train_files, sequence_len=sequence_len, cache_dir=cache_dir)
    val_ds = RolloutDataset.from_file_list(val_files, sequence_len=sequence_len, cache_dir=cache_dir)
    n_val = min(max_val_windows, len(val_ds))
    window_rng = np.random.RandomState(data_split_seed)
    indices = window_rng.choice(len(val_ds), size=n_val, replace=False).tolist()
    loader_kwargs = dict(batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
    return (
        DataLoader(train_ds, **loader_kwargs),
        DataLoader(Subset(val_ds, indices), **loader_kwargs),
        {"train_files": len(train_files), "val_files": len(val_files), "val_windows": n_val},
    )


def reward_mean(loader: DataLoader) -> float:
    """Mean target reward over the same sliding-window training distribution."""
    total, count = 0.0, 0
    for batch in loader:
        rewards = batch["reward"]
        total += rewards.sum().item()
        count += rewards.numel()
    if count == 0:
        raise RuntimeError("Cannot compute a baseline from an empty training loader")
    return total / count


def evaluate_loader(model: Any, loader: DataLoader, device: torch.device, train_reward_mean: float) -> Dict[str, float]:
    """Evaluate all B×T transitions using the canonical temporal contract."""
    model.eval()
    total_sse = total_abs = total_baseline_sse = 0.0
    transition_count = 0
    with torch.no_grad():
        for batch in loader:
            obs = batch["obs"].to(device, non_blocking=True)
            actions = batch["action"].to(device, non_blocking=True)
            rewards = batch["reward"].to(device, non_blocking=True)
            predecessor = batch["predecessor_action"].to(device, non_blocking=True)
            batch_size, sequence_len = rewards.shape
            prev_actions = torch.empty_like(actions)
            prev_actions[:, 0] = predecessor
            if sequence_len > 1:
                prev_actions[:, 1:] = actions[:, :-1]
            output = model.forward_sequence(
                obs, prev_actions, actions, force_keep_input=True,
            )
            predictions = output.reward_pred_seq
            total_sse += F.mse_loss(predictions, rewards, reduction="sum").item()
            total_abs += torch.abs(predictions - rewards).sum().item()
            total_baseline_sse += (rewards - train_reward_mean).square().sum().item()
            transition_count += batch_size * sequence_len
    if transition_count == 0:
        raise RuntimeError("Evaluation loader contains no transitions")
    return {
        "val_mse": total_sse / transition_count,
        "val_mae": total_abs / transition_count,
        "baseline_mse": total_baseline_sse / transition_count,
        "transitions": transition_count,
    }


def action_probe(model: Any, device: torch.device) -> Dict[str, Any]:
    """Action sensitivity after one model pass; report the effective policy."""
    probe_actions = [
        ("zeros", torch.tensor([[0.0, 0.0, 0.0]], device=device)),
        ("steer", torch.tensor([[1.0, 0.0, 0.0]], device=device)),
        ("gas", torch.tensor([[0.0, 1.0, 0.0]], device=device)),
        ("brake", torch.tensor([[0.0, 0.0, 1.0]], device=device)),
    ]
    image = torch.randn(1, 3, 64, 64, device=device)
    zero = torch.zeros(1, ACTION_DIM, device=device)
    with torch.no_grad():
        state = model(image, prev_action=zero, current_action=zero, force_keep_input=True).world_state
        representation = model.controller.encode(state)
        predictions = [model.controller.predict_reward(representation, action).item() for _, action in probe_actions]
    rounded = [round(value, 6) for value in predictions]
    return {"preds": rounded, "unique": len(set(rounded)), "passed": len(set(rounded)) > 1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an existing reward checkpoint without retraining.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Exact JSON output path; defaults to a non-overwriting filename beside checkpoint.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--data-split-seed", type=int, default=None, help="File/window split seed; default is checkpoint experiment seed.")
    parser.add_argument("--inference-rng-seed", type=int, default=0, help="Only controls posterior sampling and action-probe randomness.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-len", type=int, default=16)
    parser.add_argument("--max-val-windows", type=int, default=256)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--tokenizer-eval-mode", choices=["sample", "mean"], default=None)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint.get("config")
    saved_policy = _perception_value(config, "tokenizer_eval_mode", "sample")
    data_split_seed = args.data_split_seed if args.data_split_seed is not None else int(_config_value(config, "seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, data_info = build_protocol_loaders(
        args.data_root, sequence_len=args.sequence_len, batch_size=args.batch_size,
        max_val_windows=args.max_val_windows, data_split_seed=data_split_seed,
        cache_dir=args.cache_dir,
    )
    train_mean = reward_mean(train_loader)
    set_seed(args.inference_rng_seed)
    model = model_from_checkpoint(
        checkpoint, action_dim=ACTION_DIM,
        tokenizer_eval_mode_override=args.tokenizer_eval_mode,
    ).to(device)
    metrics = evaluate_loader(model, val_loader, device, train_mean)
    probe = action_probe(model, device)
    runtime_policy = model.tokenizer.eval_mode
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data_root": str(args.data_root.resolve()),
        "data_split_seed": data_split_seed,
        "inference_rng_seed": args.inference_rng_seed,
        "sequence_len": args.sequence_len,
        "batch_size": args.batch_size,
        "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else "",
        **data_info,
        "train_reward_mean": train_mean,
        **metrics,
        "ratio": metrics["val_mse"] / max(1e-8, metrics["baseline_mse"]),
        "tokenizer_policy_saved": saved_policy,
        "tokenizer_policy_override": args.tokenizer_eval_mode,
        "tokenizer_policy_runtime": runtime_policy,
        "action_probe": probe,
    }
    output = args.out or args.checkpoint.parent / (
        f"checkpoint_eval_{runtime_policy}_split{data_split_seed}_rng{args.inference_rng_seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"Refusing to overwrite existing evaluation output: {output}")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
