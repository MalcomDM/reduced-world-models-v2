#!/usr/bin/env python3
"""Evaluate S3 dropout-trained anchors: visible, masked factual, action probe."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rwm.data.rollout_dataset import RolloutDataset, _collect_npz_files
from rwm.evaluation.masked_factual_evaluator import MaskedFactualEvaluator
from rwm.utils.checkpointing import load_checkpoint, model_from_checkpoint

DATA_ROOT = Path("data/rollouts/rwm_deterministic/scenario_0")
CACHE_DIR = Path("data/cache/rollout_frames_v1")

BASE = Path("runs/component_refinement/sru_temporal/04_observational_dropout_anchor")

CKPTS = {
    42: BASE / "seed42" / "checkpoint_best.pt",
    43: BASE / "seed43" / "checkpoint_best.pt",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for seed in [42, 43]:
    ckpt_path = CKPTS[seed]
    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"Checkpoint: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path)
    config = ckpt.get("config")
    train_reward_mean = 0.0
    if config is not None and hasattr(config, "training"):
        # Read baseline from training config
        pass

    model = model_from_checkpoint(ckpt, tokenizer_eval_mode_override="mean")
    model.to(device)
    model.eval()
    print(f"  Backend: {model._temporal_backend}")

    # Build validation loader (random-burn-in SRU)
    all_files = _collect_npz_files(DATA_ROOT)
    rng = np.random.RandomState(seed)
    rng.shuffle(all_files)
    n_val = max(1, int(len(all_files) * 0.2))
    val_files = all_files[:n_val]

    val_ds = RolloutDataset.from_file_list(
        val_files, sequence_len=16, cache_dir=CACHE_DIR,
        recurrent_context=True, burn_in_steps=20,
    )
    n_windows = min(256, len(val_ds))
    rng2 = np.random.RandomState(0)
    indices = rng2.choice(len(val_ds), size=n_windows, replace=False).tolist()
    subset = Subset(val_ds, indices)
    loader = DataLoader(subset, batch_size=8, shuffle=False, num_workers=0)

    # Compute mean reward from training set for baseline
    train_files = all_files[n_val:]
    train_ds = RolloutDataset.from_file_list(
        train_files[:2], sequence_len=16, cache_dir=CACHE_DIR,
        recurrent_context=True, burn_in_steps=20,
    )
    train_rews = []
    with torch.no_grad():
        for i in range(min(50, len(train_ds))):
            s = train_ds[i]
            train_rews.append(s["reward"].mean().item())
    train_reward_mean = float(np.mean(train_rews)) if train_rews else 0.0
    print(f"  Train reward mean (est): {train_reward_mean:.4f}")

    # 1. Visible evaluation via forward_sequence
    total_mse, total_mae, total_baseline_sse, total_count = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            obs = batch["obs"].to(device)
            act = batch["action"].to(device)
            rew = batch["reward"].to(device)
            pred = batch["predecessor_action"].to(device)
            loss_mask = batch.get("loss_mask")
            if loss_mask is not None:
                loss_mask = loss_mask.to(device)
            valid_step = batch.get("valid_step")
            if valid_step is not None:
                valid_step = valid_step.to(device)

            B, T = rew.shape
            prev_actions = torch.zeros(B, T, 3, device=device)
            if T > 1:
                prev_actions[:, 1:] = act[:, :T - 1]
            if valid_step is not None:
                fv = valid_step.long().argmax(dim=1)
                for b in range(B):
                    fv_b = fv[b].item()
                    if valid_step[b, fv_b]:
                        prev_actions[b, fv_b] = pred[b]
            else:
                prev_actions[:, 0] = pred

            out = model.forward_sequence(obs, prev_actions, act, force_keep_input=True, valid_step=valid_step)
            r_pred = out.reward_pred_seq
            r_true = rew

            if loss_mask is not None:
                lm_f = loss_mask.float()
                total_mse += ((r_pred - r_true).pow(2) * lm_f).sum().item()
                total_mae += (torch.abs(r_pred - r_true) * lm_f).sum().item()
                total_baseline_sse += ((r_true - train_reward_mean).pow(2) * lm_f).sum().item()
                total_count += lm_f.sum().item()
            else:
                total_mse += torch.nn.functional.mse_loss(r_pred[:, :16], r_true[:, :16], reduction="sum").item()
                total_mae += torch.abs(r_pred[:, :16] - r_true[:, :16]).sum().item()
                total_baseline_sse += (r_true[:, :16] ** 2).sum().item()
                total_count += B * 16

    mse = total_mse / max(1, total_count)
    mae = total_mae / max(1, total_count)
    baseline_mse = total_baseline_sse / max(1, total_count)
    ratio = mse / max(1e-8, baseline_mse)
    print(f"  Visible MSE={mse:.4f} MAE={mae:.4f} ratio={ratio:.4f}")

    # 2. Masked factual evaluation
    evaluator = MaskedFactualEvaluator(model, device, train_reward_mean=train_reward_mean)
    masked_results = evaluator.evaluate(loader, warmup=4, mask_horizons=(1, 2, 4, 8, 12),
                                        action_variants=("correct", "zero", "shifted"))

    print(f"  Masked results:")
    for h in masked_results["horizons"]:
        print(f"    warmup={h['warmup']} horizon={h['mask_horizon']} variant={h['action_variant']} "
              f"MSE={h['val_mse']:.4f} ratio={h['ratio']:.4f}")

    # 3. Action probe
    with torch.no_grad():
        img = torch.randn(1, 3, 64, 64, device=device)
        base = model(img=img, prev_action=torch.zeros(1, 3, device=device),
                     current_action=torch.zeros(1, 3, device=device), force_keep_input=True)
        h = model.controller.encode(base.world_state)
        acts = [("zeros", [0, 0, 0]), ("steer", [1, 0, 0]), ("gas", [0, 1, 0]), ("brake", [0, 0, 1])]
        preds = {n: round(model.controller.predict_reward(h, torch.tensor([a], device=device)).item(), 6) for n, a in acts}
        unique = len(set(preds.values()))
        print(f"  Action probe: {unique}/4 -> {'PASS' if unique == 4 else 'FAIL'}")

    # Save
    out = {
        "seed": seed,
        "checkpoint": str(ckpt_path),
        "visible": {"mse": mse, "mae": mae, "baseline_mse": baseline_mse, "ratio": ratio},
        "masked_horizons": masked_results["horizons"],
        "action_probe": {"unique": unique, "passed": unique == 4, "predictions": preds},
    }
    (BASE / f"evaluation_seed{seed}.json").write_text(json.dumps(out, indent=2))
    print(f"  Saved to evaluation_seed{seed}.json")
