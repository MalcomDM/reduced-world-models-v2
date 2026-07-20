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
from rwm.data.split import collect_and_split
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


def build_protocol_loaders(
    root: Path,
    *,
    sequence_len: int,
    batch_size: int,
    max_val_windows: int,
    data_split_seed: int,
    cache_dir: Optional[Path],
    recurrent_context: bool = False,
    burn_in_steps: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Build fixed train/validation loaders without consuming inference RNG."""
    train_files, val_files = collect_and_split(root, data_split_seed)
    train_ds = RolloutDataset.from_file_list(
        train_files, sequence_len=sequence_len, cache_dir=cache_dir,
        recurrent_context=recurrent_context, burn_in_steps=burn_in_steps,
    )
    val_ds = RolloutDataset.from_file_list(
        val_files, sequence_len=sequence_len, cache_dir=cache_dir,
        recurrent_context=recurrent_context, burn_in_steps=burn_in_steps,
    )
    n_val = min(max_val_windows, len(val_ds))
    window_rng = np.random.RandomState(data_split_seed)
    indices = window_rng.choice(len(val_ds), size=n_val, replace=False).tolist()
    loader_kwargs = dict(batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
    return (
        DataLoader(train_ds, **loader_kwargs),
        DataLoader(Subset(val_ds, indices), **loader_kwargs),
        {
            "train_files": len(train_files), "val_files": len(val_files),
            "val_windows": n_val, "recurrent_context": recurrent_context,
            "burn_in_steps": burn_in_steps,
        },
    )


def reward_mean(loader: DataLoader, eval_mode: str = "canonical") -> float:
    """Mean reward over the same mask used by the requested evaluation."""
    total, count = 0.0, 0
    for batch in loader:
        rewards = batch["reward"]
        loss_mask = batch.get("loss_mask")
        valid_step = batch.get("valid_step")
        mask = _build_eval_mask(
            loss_mask, valid_step, rewards.shape[1], eval_mode, rewards.device,
        )
        if mask.shape != rewards.shape:
            mask = mask.expand_as(rewards)
        total += (rewards * mask).sum().item()
        count += int(mask.sum().item())
    if count == 0:
        raise RuntimeError("Cannot compute a baseline from an empty training loader")
    return total / count


def _build_eval_mask(loss_mask: torch.Tensor, valid_step: torch.Tensor,
                     sequence_len: int, eval_mode: str, device: torch.device) -> torch.Tensor:
    """Build evaluation mask for the requested mode."""
    if loss_mask is not None:
        loss_mask = loss_mask.to(device)
    if valid_step is not None:
        valid_step = valid_step.to(device)

    if eval_mode == "canonical":
        if loss_mask is not None:
            return loss_mask
        return torch.ones(1, sequence_len, dtype=torch.bool, device=device)

    if eval_mode == "all_36":
        if valid_step is not None:
            return valid_step
        return torch.ones(1, sequence_len, dtype=torch.bool, device=device)

    if eval_mode == "tail_16":
        # Process all positions but score only the final 16.
        if valid_step is not None:
            # Find the valid region: [first_valid, last_valid]
            B = valid_step.shape[0]
            T = valid_step.shape[1]
            mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            if T < 20:
                return mask
            # tail_16: positions [T-16, T) within the valid region.
            for b in range(B):
                last_valid = int(valid_step[b].nonzero(as_tuple=True)[0].max().item()) + 1 if valid_step[b].any() else 0
                tail_start = max(0, last_valid - 16)
                mask[b, tail_start:last_valid] = True
            return mask
        # No valid_step: assume all positions are valid.
        T = sequence_len
        mask = torch.zeros(1, T, dtype=torch.bool, device=device)
        tail_start = max(0, T - 16)
        mask[:, tail_start:] = True
        return mask

    raise ValueError(f"Unknown evaluation mode: {eval_mode}")


def evaluate_loader(model: Any, loader: DataLoader, device: torch.device,
                    train_reward_mean: float, eval_mode: str = "canonical") -> Dict[str, float]:
    """Evaluate transitions using the chosen evaluation mask."""
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
            valid_step = batch.get("valid_step")
            loss_mask = batch.get("loss_mask")
            if valid_step is not None:
                valid_step = valid_step.to(device, non_blocking=True)
                first_valid = valid_step.long().argmax(dim=1)
                for b in range(batch_size):
                    index = first_valid[b].item()
                    if valid_step[b, index]:
                        prev_actions[b, index] = predecessor[b]
            sequence_kwargs = {"force_keep_input": True}
            if valid_step is not None:
                sequence_kwargs["valid_step"] = valid_step
            output = model.forward_sequence(obs, prev_actions, actions, **sequence_kwargs)
            predictions = output.reward_pred_seq

            # Build evaluation mask.
            mask = _build_eval_mask(loss_mask, valid_step, sequence_len, eval_mode, device)
            if mask.shape != rewards.shape:
                mask = mask.expand_as(rewards)
            total_sse += ((predictions - rewards).square() * mask).sum().item()
            total_abs += (torch.abs(predictions - rewards) * mask).sum().item()
            total_baseline_sse += ((rewards - train_reward_mean).square() * mask).sum().item()
            transition_count += int(mask.sum().item())

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
    parser.add_argument(
        "--sequence-len", type=int, default=None,
        help="Evaluation sequence length; defaults to the checkpoint data/temporal config.",
    )
    parser.add_argument("--max-val-windows", type=int, default=256)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--tokenizer-eval-mode", choices=["sample", "mean"], default=None)
    parser.add_argument("--evaluation-mode", type=str, default="canonical",
                        choices=["canonical", "all_36", "tail_16"],
                        help="Evaluation mask: canonical=loss_mask, all_36=all valid, tail_16=positions 20:36")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint.get("config")
    saved_policy = _perception_value(config, "tokenizer_eval_mode", "sample")
    data_split_seed = args.data_split_seed if args.data_split_seed is not None else int(_config_value(config, "seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temporal = _config_value(config, "temporal", None)
    backend = _config_value(temporal, "backend", "causal_transformer")
    data_config = _config_value(config, "data", None)
    saved_sequence_len = int(
        _config_value(
            data_config,
            "sequence_len",
            _config_value(temporal, "seq_len", 16),
        )
    )
    sequence_len = args.sequence_len if args.sequence_len is not None else saved_sequence_len
    recurrent_context = backend == "minimal_sru"
    burn_in_steps = int(_config_value(temporal, "sru_burn_in_steps", 0)) if recurrent_context else 0
    train_loader, val_loader, data_info = build_protocol_loaders(
        args.data_root, sequence_len=sequence_len, batch_size=args.batch_size,
        max_val_windows=args.max_val_windows, data_split_seed=data_split_seed,
        cache_dir=args.cache_dir, recurrent_context=recurrent_context,
        burn_in_steps=burn_in_steps,
    )
    train_mean = reward_mean(train_loader, eval_mode=args.evaluation_mode)
    set_seed(args.inference_rng_seed)
    model = model_from_checkpoint(
        checkpoint, action_dim=ACTION_DIM,
        tokenizer_eval_mode_override=args.tokenizer_eval_mode,
    ).to(device)
    metrics = evaluate_loader(model, val_loader, device, train_mean, eval_mode=args.evaluation_mode)
    probe = action_probe(model, device)
    runtime_policy = model.tokenizer.eval_mode
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data_root": str(args.data_root.resolve()),
        "data_split_seed": data_split_seed,
        "inference_rng_seed": args.inference_rng_seed,
        "sequence_len": sequence_len,
        "sequence_len_saved": saved_sequence_len,
        "batch_size": args.batch_size,
        "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else "",
        **data_info,
        "train_reward_mean": train_mean,
        **metrics,
        "ratio": metrics["val_mse"] / max(1e-8, metrics["baseline_mse"]),
        "tokenizer_policy_saved": saved_policy,
        "tokenizer_policy_override": args.tokenizer_eval_mode,
        "tokenizer_policy_runtime": runtime_policy,
        "temporal_backend": backend,
        "evaluation_mode": args.evaluation_mode,
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
