#!/usr/bin/env python3
"""Resume SRU-4 from checkpoint_latest for 7 additional epochs, matching
approximate SRU-20 wall-clock time.

Usage:
    python scripts/experiments/sru/run_sru4_matched_wallclock.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rwm.config.experiment_config import ExperimentConfig, TemporalConfig
from rwm.data.rollout_dataset import RolloutDataset, _collect_npz_files
from rwm.trainers.deterministic.world_model_trainer import WorldModelTrainer
from rwm.utils.checkpointing import load_checkpoint, save_checkpoint
from rwm.utils.seeding import set_seed


DATA_ROOT = Path("data/rollouts/rwm_deterministic/scenario_0")
CACHE_DIR = Path("data/cache/rollout_frames_v1")
OUT_BASE = Path("runs/component_refinement/sru_temporal/06_sru4_matched_wallclock")
PARENT_BASE = Path("runs/component_refinement/sru_temporal/05_canonical_burnin_comparison")
SRU20_BASE = Path("runs/component_refinement/sru_temporal/05_canonical_burnin_comparison")
ADDITIONAL_EPOCHS = 7
BATCH_SIZE = 8
SEQUENCE_LEN = 16
VAL_WINDOWS = 256


def resume_sru4(seed: int):
    """Resume SRU-4 checkpoint_latest for 7 additional epochs."""
    print(f"\n{'='*60}")
    print(f"Resuming SRU-4 seed {seed}")

    parent_dir = PARENT_BASE / f"sru4_seed{seed}"
    ckpt_path = parent_dir / "checkpoint_latest.pt"
    if not ckpt_path.exists():
        print(f"  ERROR: {ckpt_path} not found")
        return

    out_dir = OUT_BASE / f"sru4_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {out_dir}")

    # Load checkpoint.
    ckpt = load_checkpoint(ckpt_path)
    cfg_dict = ckpt.get("config")
    assert cfg_dict is not None, "Checkpoint must have config"
    if hasattr(cfg_dict, "to_dict"):
        cfg_dict = cfg_dict.to_dict()
    cfg = ExperimentConfig.from_dict(cfg_dict)
    print(f"  Config backend: {cfg.temporal.backend}, burn_in: {cfg.temporal.sru_burn_in_steps}")
    assert cfg.temporal.backend == "minimal_sru"
    assert cfg.temporal.sru_burn_in_steps == 4
    assert cfg.temporal.sru_training_mode == "random_burn_in"

    # Restore seed.
    set_seed(seed)

    # Build dataset (same split as parent).
    all_files = _collect_npz_files(DATA_ROOT)
    rng = np.random.RandomState(seed)
    rng.shuffle(all_files)
    n_val = max(1, int(len(all_files) * 0.2))
    train_files = all_files[n_val:]
    val_files = all_files[:n_val]

    is_sru = True
    train_ds = RolloutDataset.from_file_list(
        train_files, sequence_len=SEQUENCE_LEN, cache_dir=CACHE_DIR,
        recurrent_context=True, burn_in_steps=4,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=0)

    val_ds = RolloutDataset.from_file_list(
        val_files, sequence_len=SEQUENCE_LEN, cache_dir=CACHE_DIR,
        recurrent_context=True, burn_in_steps=4,
    )
    n_val_win = min(VAL_WINDOWS, len(val_ds))
    rng2 = np.random.RandomState(0)
    indices = rng2.choice(len(val_ds), size=n_val_win, replace=False).tolist()
    val_subset = Subset(val_ds, indices)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Build model and load weights.
    from rwm.models.rwm.model import ReducedWorldModel
    from rwm.config.experiment_config import PerceptionConfig, ControllerConfig, TrainingConfig, TemporalMaskConfig

    model = ReducedWorldModel(
        action_dim=3,
        reward_head_kind=getattr(cfg.controller, "reward_head_kind", "linear"),
        reward_head_hidden_dim=getattr(cfg.controller, "reward_head_hidden_dim", 32),
        selection_mode=getattr(cfg.perception, "selection_mode", "learned"),
        selection_k=getattr(cfg.perception, "k", 8),
        selection_seed=getattr(cfg.perception, "selection_seed", 0),
        tokenizer_eval_mode=getattr(cfg.perception, "tokenizer_eval_mode", "sample"),
        temporal_config=cfg.temporal,
    )
    model.load_state_dict(ckpt["model_state"])
    print(f"  Model weights restored.")

    # Restore optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    if ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"  Optimizer state restored.")

    # Build trainer wrapping the restored components.
    trainer = WorldModelTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        out_dir=out_dir,
        sequence_len=SEQUENCE_LEN,
        epochs=ADDITIONAL_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=cfg.training.learning_rate,
        beta=cfg.training.beta,
        config=cfg,
    )
    # Overwrite the model and optimizer created by __init__.
    trainer.model = model.to(trainer.device)
    trainer.optimizer = optimizer
    for state in trainer.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(trainer.device)

    print(f"  Trainer ready. {ADDITIONAL_EPOCHS} additional epochs.")

    # Read parent metrics for accounting.
    parent_metrics_path = parent_dir / "metrics.csv"
    parent_metrics = {}
    if parent_metrics_path.exists():
        lines = parent_metrics_path.read_text().strip().split("\n")
        if len(lines) >= 2:
            keys = lines[0].split(",")
            parent_vals = lines[-1].split(",")
            parent_metrics = dict(zip(keys, parent_vals))

    # Record parent train time.
    parent_results_path = parent_dir / "results.json"
    parent_train_time = 0.0
    if parent_results_path.exists():
        pr = json.loads(parent_results_path.read_text())
        parent_train_time = pr.get("elapsed_s", 0)

    # Train additional epochs, validate only at the end.
    start = time.time()
    trainer.fit(validate_every=ADDITIONAL_EPOCHS, progress_label="Add epoch",
                reference_targets_per_epoch=len(train_ds) * 16)
    added_time = time.time() - start

    # Save final checkpoint with full config.
    final_state = trainer.model.state_dict()
    save_checkpoint(
        path=out_dir / "checkpoint_latest",
        model_state=final_state,
        config=cfg,
        optimizer_state=trainer.optimizer.state_dict(),
        global_step=trainer._global_step,
        epoch=10 + ADDITIONAL_EPOCHS,
        metrics={"train_mse": trainer._last_train_reward_mean},
    )

    # Read added metrics.
    added_metrics_path = out_dir / "metrics.csv"
    added_metrics = {}
    if added_metrics_path.exists():
        lines = added_metrics_path.read_text().strip().split("\n")
        if len(lines) >= 2:
            keys = lines[0].split(",")
            added_vals = lines[-1].split(",")
            added_metrics = dict(zip(keys, added_vals))

    # Frozen evaluation.
    trainer.model.eval()
    dev = trainer.device
    total_mse, total_abs, total_baseline, total_count = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            obs = batch["obs"].to(dev)
            act = batch["action"].to(dev)
            rew = batch["reward"].to(dev)
            pred = batch["predecessor_action"].to(dev)
            loss_mask = batch.get("loss_mask")
            if loss_mask is not None:
                loss_mask = loss_mask.to(dev)
            valid_step = batch.get("valid_step")
            if valid_step is not None:
                valid_step = valid_step.to(dev)
            B, T = rew.shape
            prev_acts = torch.zeros(B, T, 3, device=dev)
            if T > 1:
                prev_acts[:, 1:] = act[:, :T - 1]
            if valid_step is not None:
                fv = valid_step.long().argmax(dim=1)
                for b_idx in range(B):
                    fv_b = fv[b_idx].item()
                    if valid_step[b_idx, fv_b]:
                        prev_acts[b_idx, fv_b] = pred[b_idx]
            else:
                prev_acts[:, 0] = pred
            out = trainer.model.forward_sequence(obs, prev_acts, act, force_keep_input=True, valid_step=valid_step)
            rp = out.reward_pred_seq
            if loss_mask is not None:
                lm_f = loss_mask.float()
                total_mse += ((rp - rew).pow(2) * lm_f).sum().item()
                total_abs += (torch.abs(rp - rew) * lm_f).sum().item()
                total_baseline += (rew.pow(2) * lm_f).sum().item()
                total_count += lm_f.sum().item()
            else:
                total_mse += ((rp[:, :SEQUENCE_LEN] - rew[:, :SEQUENCE_LEN]) ** 2).sum().item()
                total_abs += (torch.abs(rp[:, :SEQUENCE_LEN] - rew[:, :SEQUENCE_LEN])).sum().item()
                total_baseline += (rew[:, :SEQUENCE_LEN] ** 2).sum().item()
                total_count += B * SEQUENCE_LEN

    frozen_mse = total_mse / max(1, total_count)
    frozen_baseline = total_baseline / max(1, total_count)
    frozen_ratio = frozen_mse / max(1e-8, frozen_baseline)

    # Action probe.
    with torch.no_grad():
        img = torch.randn(1, 3, 64, 64, device=dev)
        base = trainer.model(img=img, prev_action=torch.zeros(1, 3, device=dev),
                             current_action=torch.zeros(1, 3, device=dev), force_keep_input=True)
        h = trainer.model.controller.encode(base.world_state)
        acts = [torch.tensor([a], device=dev) for a in [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]]
        preds = [trainer.model.controller.predict_reward(h, a).item() for a in acts]
        unique = len(set(round(p, 6) for p in preds))

    results = {
        "seed": seed,
        "parent_checkpoint": str(ckpt_path),
        "parent_epochs": 10,
        "added_epochs": ADDITIONAL_EPOCHS,
        "total_epochs": 10 + ADDITIONAL_EPOCHS,
        "parent_train_time_s": parent_train_time,
        "added_train_time_s": round(added_time, 2),
        "total_wall_s": round(parent_train_time + added_time, 2),
        "added_opt_updates": int(added_metrics.get("opt_updates", 0)),
        "parent_opt_updates": int(parent_metrics.get("opt_updates", 0)),
        "total_opt_updates": int(parent_metrics.get("opt_updates", 0)) + int(added_metrics.get("opt_updates", 0)),
        "added_target_transitions": int(added_metrics.get("real_target_transitions", 0)),
        "parent_target_transitions": int(parent_metrics.get("real_target_transitions", 0)),
        "total_target_transitions": int(parent_metrics.get("real_target_transitions", 0)) + int(added_metrics.get("real_target_transitions", 0)),
        "added_model_positions": int(added_metrics.get("processed_model_positions", 0)),
        "parent_model_positions": int(parent_metrics.get("processed_model_positions", 0)),
        "total_model_positions": int(parent_metrics.get("processed_model_positions", 0)) + int(added_metrics.get("processed_model_positions", 0)),
        "frozen_val_mse": round(frozen_mse, 6),
        "frozen_val_ratio": round(frozen_ratio, 4),
        "action_probe_unique": unique,
        "action_probe_passed": unique == 4,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"  Added time: {added_time:.1f}s, total: {parent_train_time + added_time:.0f}s")
    print(f"  Total opt updates: {results['total_opt_updates']}")
    print(f"  Total target transitions: {results['total_target_transitions']}")
    print(f"  Frozen ratio: {frozen_ratio:.4f}")
    print(f"  Action probe: {unique}/4")
    print(f"  Results saved.")
    return results


def eval_sru20_latest(seed: int):
    """Evaluate SRU-20 checkpoint_latest with frozen evaluator."""
    print(f"\n--- Evaluating SRU-20 seed {seed} checkpoint_latest ---")
    ckpt_path = SRU20_BASE / f"sru20_seed{seed}" / "checkpoint_latest.pt"
    if not ckpt_path.exists():
        print(f"  ERROR: {ckpt_path} not found")
        return {}

    ckpt = load_checkpoint(ckpt_path)
    cfg_dict = ckpt.get("config")
    if hasattr(cfg_dict, "to_dict"):
        cfg_dict = cfg_dict.to_dict()
    cfg = ExperimentConfig.from_dict(cfg_dict)

    from rwm.models.rwm.model import ReducedWorldModel
    model = ReducedWorldModel(
        action_dim=3,
        reward_head_kind=getattr(cfg.controller, "reward_head_kind", "linear"),
        reward_head_hidden_dim=getattr(cfg.controller, "reward_head_hidden_dim", 32),
        selection_mode=getattr(cfg.perception, "selection_mode", "learned"),
        selection_k=getattr(cfg.perception, "k", 8),
        selection_seed=getattr(cfg.perception, "selection_seed", 0),
        tokenizer_eval_mode="mean",
        temporal_config=cfg.temporal,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_files = _collect_npz_files(DATA_ROOT)
    rng = np.random.RandomState(seed)
    rng.shuffle(all_files)
    n_val = max(1, int(len(all_files) * 0.2))
    val_files = all_files[:n_val]

    val_ds = RolloutDataset.from_file_list(
        val_files, sequence_len=SEQUENCE_LEN, cache_dir=CACHE_DIR,
        recurrent_context=True, burn_in_steps=20,
    )
    n_val_win = min(VAL_WINDOWS, len(val_ds))
    rng2 = np.random.RandomState(0)
    indices = rng2.choice(len(val_ds), size=n_val_win, replace=False).tolist()
    subset = Subset(val_ds, indices)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    total_mse, total_abs, total_baseline, total_count = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            obs = batch["obs"].to(device)
            act = batch["action"].to(device)
            rew = batch["reward"].to(device)
            pred = batch["predecessor_action"].to(device)
            loss_mask = batch["loss_mask"].to(device) if "loss_mask" in batch else None
            valid_step = batch["valid_step"].to(device) if "valid_step" in batch else None
            B, T = rew.shape
            prev_acts = torch.zeros(B, T, 3, device=device)
            if T > 1:
                prev_acts[:, 1:] = act[:, :T - 1]
            if valid_step is not None:
                fv = valid_step.long().argmax(dim=1)
                for b_idx in range(B):
                    fv_b = fv[b_idx].item()
                    if valid_step[b_idx, fv_b]:
                        prev_acts[b_idx, fv_b] = pred[b_idx]
            else:
                prev_acts[:, 0] = pred
            out = model.forward_sequence(obs, prev_acts, act, force_keep_input=True, valid_step=valid_step)
            rp = out.reward_pred_seq
            if loss_mask is not None:
                lm_f = loss_mask.float()
                total_mse += ((rp - rew).pow(2) * lm_f).sum().item()
                total_abs += (torch.abs(rp - rew) * lm_f).sum().item()
                total_baseline += (rew.pow(2) * lm_f).sum().item()
                total_count += lm_f.sum().item()
            else:
                total_mse += ((rp[:, :SEQUENCE_LEN] - rew[:, :SEQUENCE_LEN]) ** 2).sum().item()
                total_abs += (torch.abs(rp[:, :SEQUENCE_LEN] - rew[:, :SEQUENCE_LEN])).sum().item()
                total_baseline += (rew[:, :SEQUENCE_LEN] ** 2).sum().item()
                total_count += B * SEQUENCE_LEN

    mse = total_mse / max(1, total_count)
    baseline = total_baseline / max(1, total_count)
    ratio = mse / max(1e-8, baseline)

    # Action probe.
    with torch.no_grad():
        img = torch.randn(1, 3, 64, 64, device=device)
        base = model(img=img, prev_action=torch.zeros(1, 3, device=device),
                     current_action=torch.zeros(1, 3, device=device), force_keep_input=True)
        h = model.controller.encode(base.world_state)
        acts = [torch.tensor([a], device=device) for a in [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]]
        preds = [model.controller.predict_reward(h, a).item() for a in acts]
        unique = len(set(round(p, 6) for p in preds))

    result = {
        "seed": seed,
        "source": "checkpoint_latest (epoch 10)",
        "frozen_mse": round(mse, 6),
        "frozen_ratio": round(ratio, 4),
        "action_probe": f"{unique}/4",
    }
    print(f"  Frozen MSE={mse:.4f} ratio={ratio:.4f} probe={unique}/4")
    return result


def main():
    print("SRU-4 Matched Wall-Clock Optimization Probe")
    print(f"Resuming from checkpoint_latest for {ADDITIONAL_EPOCHS} additional epochs")
    print(f"Output: {OUT_BASE}")

    # Run seeds sequentially.
    r42 = resume_sru4(42)
    r43 = resume_sru4(43)

    # Re-evaluate SRU-20 checkpoint_latest.
    s20_42 = eval_sru20_latest(42)
    s20_43 = eval_sru20_latest(43)

    # Side-by-side comparison.
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON (latest state, not best checkpoint)")
    print(f"{'='*70}")
    print(f"{'Metric':<45} {'SRU-4 seed42':<18} {'SRU-20 seed42':<18}")
    print(f"{'-'*81}")
    for key in ["frozen_val_ratio", "total_wall_s", "total_opt_updates",
                "total_target_transitions", "total_model_positions"]:
        v4 = r42.get(key, "?") if r42 else "?"
        v20 = s20_42.get("frozen_ratio" if key == "frozen_val_ratio" else key, "?")
        if isinstance(v20, str):
            v20 = s20_42.get({"frozen_val_ratio": "frozen_ratio"}.get(key, key), "?")
        print(f"  {key:<45} {str(v4):<18} {str(v20):<18}")

    print(f"{'='*70}")
    print(f"{'Metric':<45} {'SRU-4 seed43':<18} {'SRU-20 seed43':<18}")
    print(f"{'-'*81}")
    for key in ["frozen_val_ratio", "total_wall_s", "total_opt_updates",
                "total_target_transitions", "total_model_positions"]:
        v4 = r43.get(key, "?") if r43 else "?"
        v20 = s20_43.get("frozen_ratio" if key == "frozen_val_ratio" else key, "?")
        if isinstance(v20, str):
            v20 = s20_43.get({"frozen_val_ratio": "frozen_ratio"}.get(key, key), "?")
        print(f"  {key:<45} {str(v4):<18} {str(v20):<18}")

    # Statement.
    print(f"\n")
    print(f"SRU-4 vs SRU-20 conclusion under approximate matched wall time:")
    print(f"  SRU-4 receives more optimizer updates and target exposures")
    print(f"  but less context per update (4-step burn-in vs 20-step).")
    print(f"  The quality gap is reported in RESULTS.md")


if __name__ == "__main__":
    main()
