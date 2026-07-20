#!/usr/bin/env python3
"""Preflight smoke for S4B canonical burn-in comparison.

Tests every backend/burn-in combination with one tiny epoch:
  - config persistence
  - checkpoint reload
  - evaluator invocation
  - metric-schema validation

Usage:
    python scripts/experiments/sru/preflight_canonical_burnin.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rwm.data.rollout_dataset import RolloutDataset, _collect_npz_files
from rwm.config.experiment_config import ExperimentConfig, TemporalConfig
from rwm.utils.checkpointing import load_checkpoint, model_from_checkpoint
from rwm.utils.seeding import set_seed

DATA_ROOT = Path("data/rollouts/rwm_deterministic/scenario_0")
CACHE_DIR = Path("data/cache/rollout_frames_v1")
OUT_BASE = Path("runs/component_refinement/sru_temporal/05_canonical_burnin_comparison/smoke")
VAL_WINDOWS = 32
BATCH_SIZE = 8
SEQUENCE_LEN = 16


def smoke(backend: str, burn_in: int, seed: int):
    variant = "causal" if backend == "causal_transformer" else f"sru{burn_in}"
    out_dir = OUT_BASE / f"{variant}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Smoke: {variant} seed {seed}")
    print(f"  backend={backend}, burn_in={burn_in}")
    print(f"  out_dir={out_dir}")

    set_seed(seed)

    is_sru = backend == "minimal_sru"

    # Build resolved config.
    cfg = ExperimentConfig(
        experiment_name=f"smoke_{variant}_{seed}",
        seed=seed,
        temporal=TemporalConfig(
            backend=backend,
            seq_len=SEQUENCE_LEN,
            sru_burn_in_steps=burn_in,
            sru_training_mode="random_burn_in",
        ),
    )

    # Save config.
    cfg.save(str(out_dir / "config.json"))
    loaded_cfg = ExperimentConfig.load(str(out_dir / "config.json"))
    assert loaded_cfg.temporal.backend == backend
    if is_sru:
        assert loaded_cfg.temporal.sru_burn_in_steps == burn_in, \
            f"Expected burn_in={burn_in}, got {loaded_cfg.temporal.sru_burn_in_steps}"
    assert loaded_cfg.temporal.sru_training_mode == "random_burn_in"
    print(f"  config: OK")

    # Build dataset.
    all_files = _collect_npz_files(DATA_ROOT)
    rng = np.random.RandomState(seed)
    rng.shuffle(all_files)
    n_val = max(1, int(len(all_files) * 0.2))
    train_files = all_files[n_val:]
    val_files = all_files[:n_val]

    train_ds = RolloutDataset.from_file_list(
        train_files[:2], sequence_len=SEQUENCE_LEN, cache_dir=CACHE_DIR,
        recurrent_context=is_sru, burn_in_steps=burn_in if is_sru else 0,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)

    val_ds = RolloutDataset.from_file_list(
        val_files, sequence_len=SEQUENCE_LEN, cache_dir=CACHE_DIR,
        recurrent_context=is_sru, burn_in_steps=burn_in if is_sru else 0,
    )
    n_val_win = min(VAL_WINDOWS, len(val_ds))
    rng2 = np.random.RandomState(0)
    indices = rng2.choice(len(val_ds), size=n_val_win, replace=False).tolist()
    val_subset = Subset(val_ds, indices)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  train windows: {len(train_ds)}, val windows: {n_val_win}")

    # Build model.
    from rwm.trainers.deterministic.world_model_trainer import WorldModelTrainer
    trainer = WorldModelTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        out_dir=out_dir,
        sequence_len=SEQUENCE_LEN,
        epochs=1,
        batch_size=BATCH_SIZE,
        config=cfg,
    )

    # Train one batch.
    loss_total, loss_mse, loss_kl, elapsed = trainer.train_one_epoch()
    assert torch.isfinite(torch.tensor(loss_total)), f"loss_total not finite: {loss_total}"
    assert torch.isfinite(torch.tensor(loss_mse)), f"loss_mse not finite: {loss_mse}"
    print(f"  train: loss={loss_total:.4f} mse={loss_mse:.4f} kl={loss_kl:.4f} ({elapsed:.2f}s)")

    # Evaluate.
    val_metrics = trainer.evaluate()
    assert "val_mse" in val_metrics
    assert torch.isfinite(torch.tensor(val_metrics["val_mse"]))
    print(f"  val: mse={val_metrics['val_mse']:.4f}")

    # Checkpoint.
    trainer.log_and_checkpoint(1, {
        "epoch": 1, "train_total": loss_total, "train_mse": loss_mse,
        "train_kl": loss_kl, "val_mse": val_metrics["val_mse"],
        "val_mae": val_metrics.get("val_mae", 0),
        "baseline_mse": val_metrics.get("mean_baseline_mse", 0), "time": elapsed,
    })

    ckpt_best = out_dir / "checkpoint_best.pt"
    assert ckpt_best.exists(), f"checkpoint_best not found"
    print(f"  checkpoint_best: OK")

    # Reload checkpoint.
    loaded_ckpt = load_checkpoint(ckpt_best)
    rebuilt = model_from_checkpoint(loaded_ckpt, tokenizer_eval_mode_override="mean")
    assert rebuilt._temporal_backend == backend
    if is_sru:
        assert isinstance(rebuilt.world_hd, type(trainer.model.world_hd))
    print(f"  checkpoint reload: OK")

    # Action probe.
    rebuilt.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rebuilt.to(device)
    with torch.no_grad():
        img = torch.randn(1, 3, 64, 64, device=device)
        base = rebuilt(img=img, prev_action=torch.zeros(1, 3, device=device),
                       current_action=torch.zeros(1, 3, device=device), force_keep_input=True)
        h = rebuilt.controller.encode(base.world_state)
        acts = [torch.tensor([a], device=device) for a in [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]]
        preds = [rebuilt.controller.predict_reward(h, a).item() for a in acts]
        unique = len(set(round(p, 6) for p in preds))
        assert unique == 4, f"Action probe: {unique}/4"
    print(f"  action probe: {unique}/4 OK")

    # Metric schema validation.
    metrics_file = out_dir / "metrics.csv"
    assert metrics_file.exists()
    with open(metrics_file) as f:
        lines = f.readlines()
    assert len(lines) >= 2  # header + at least 1 row
    header = lines[0].strip().split(",")
    required = ["epoch", "train_total", "train_mse", "train_kl", "val_mse", "time"]
    for key in required:
        assert key in header, f"Missing metric column: {key}"
    # Check that all rows have the same column count.
    n_cols = len(header)
    for i, line in enumerate(lines[1:], 1):
        cols = line.strip().split(",")
        assert len(cols) == n_cols, f"Row {i}: expected {n_cols} columns, got {len(cols)}"
    print(f"  metrics schema: OK ({n_cols} columns)")

    # Frozen evaluator.
    val_ds2 = RolloutDataset.from_file_list(
        val_files, sequence_len=SEQUENCE_LEN, cache_dir=CACHE_DIR,
        recurrent_context=is_sru, burn_in_steps=burn_in if is_sru else 0,
    )
    rng3 = np.random.RandomState(0)
    indices2 = rng3.choice(len(val_ds2), size=n_val_win, replace=False).tolist()
    subset2 = Subset(val_ds2, indices2)
    loader2 = DataLoader(subset2, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    total_mse, total_baseline, total_count = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader2:
            obs = batch["obs"].to(device)
            act = batch["action"].to(device)
            rew = batch["reward"].to(device)
            if is_sru:
                pred = batch["predecessor_action"].to(device)
                loss_mask = batch.get("loss_mask")
                if loss_mask is not None:
                    loss_mask = loss_mask.to(device)
                valid_step = batch.get("valid_step")
                if valid_step is not None:
                    valid_step = valid_step.to(device)
                B, T = rew.shape
                prev_acts = torch.zeros(B, T, 3, device=device)
                if T > 1:
                    prev_acts[:, 1:] = act[:, :T - 1]
                if valid_step is not None:
                    fv = valid_step.long().argmax(dim=1)
                    for b in range(B):
                        fv_b = fv[b].item()
                        if valid_step[b, fv_b]:
                            prev_acts[b, fv_b] = pred[b]
                else:
                    prev_acts[:, 0] = pred
                out = rebuilt.forward_sequence(obs, prev_acts, act, force_keep_input=True, valid_step=valid_step)
                rp = out.reward_pred_seq
                if loss_mask is not None:
                    lm_f = loss_mask.float()
                    total_mse += ((rp - rew).pow(2) * lm_f).sum().item()
                    total_baseline += (rew.pow(2) * lm_f).sum().item()
                    total_count += lm_f.sum().item()
                else:
                    total_mse += ((rp[:, :SEQUENCE_LEN] - rew[:, :SEQUENCE_LEN]) ** 2).sum().item()
                    total_baseline += (rew[:, :SEQUENCE_LEN] ** 2).sum().item()
                    total_count += B * SEQUENCE_LEN
            else:
                pred = batch["predecessor_action"].to(device)
                B, T = rew.shape
                seq_len = min(SEQUENCE_LEN, T)
                prev_acts = torch.zeros(B, seq_len, 3, device=device)
                prev_acts[:, 0] = pred
                if seq_len > 1:
                    prev_acts[:, 1:] = act[:, :seq_len - 1]
                out = rebuilt.forward_sequence(obs[:, :seq_len], prev_acts, act[:, :seq_len], force_keep_input=True)
                total_mse += ((out.reward_pred_seq - rew[:, :seq_len]) ** 2).sum().item()
                total_baseline += (rew[:, :seq_len] ** 2).sum().item()
                total_count += B * seq_len

    mse = total_mse / max(1, total_count)
    baseline = total_baseline / max(1, total_count)
    ratio = mse / max(1e-8, baseline)
    print(f"  frozen eval: MSE={mse:.4f} baseline={baseline:.4f} ratio={ratio:.4f}")

    print(f"  ✅ PASS")
    return True


def main():
    combos = [
        ("causal_transformer", 0, 42),
        ("causal_transformer", 0, 43),
        ("minimal_sru", 20, 42),
        ("minimal_sru", 20, 43),
        ("minimal_sru", 8, 42),
        ("minimal_sru", 8, 43),
        ("minimal_sru", 4, 42),
        ("minimal_sru", 4, 43),
        ("minimal_sru", 0, 42),
        ("minimal_sru", 0, 43),
    ]

    passed = 0
    failed = 0
    for backend, burn_in, seed in combos:
        try:
            smoke(backend, burn_in, seed)
            passed += 1
        except Exception as e:
            print(f"\n❌ FAIL: {backend} burn_in={burn_in} seed={seed}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Preflight complete: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
