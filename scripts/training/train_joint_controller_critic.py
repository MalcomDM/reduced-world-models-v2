#!/usr/bin/env python3
"""Stage 6.1 — ControllerTrunk + OnlineCritic joint training.

The first deliberately narrow end-to-end step.
Factual reward losses and imagined Critic pressure may update the
ControllerTrunk, while perception, MinimalSRU, Actor, and TargetCritic
remain frozen.

Uses a deterministic file-level train/validation split (``data_split_seed``)
independent of the training RNG (``training_seed``).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rwm.config.config import ACTION_DIM
from rwm.data.rollout_dataset import RolloutDataset
from rwm.data.split import collect_and_split
from rwm.trainers.imagined_actor_critic import (
    ImaginedACTrainer,
    ImaginedACTrainingConfig,
    load_ac_from_checkpoint,
    validate_ac_anchor_checkpoint,
)
from rwm.trainers.joint_controller_critic import (
    JointControllerCriticConfig,
    JointControllerCriticTrainer,
)
from rwm.utils.checkpointing import load_checkpoint, model_from_checkpoint, save_checkpoint


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 6.1 — ControllerTrunk + Critic joint training",
    )
    parser.add_argument("--anchor", required=True, help="Frozen world-model anchor (.pt)")
    parser.add_argument("--ac", required=True, help="Frozen Actor-Critic checkpoint (.pt)")
    parser.add_argument("--out", required=True, help="Output directory (must not exist)")
    parser.add_argument("--updates", type=int, default=500, help="Number of optimizer steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-split-seed", type=int, default=42,
                        help="RNG seed for file partition (independent of training)")
    parser.add_argument("--training-seed", type=int, default=42,
                        help="RNG seed for initialization, DataLoader order, stochastic ops")
    parser.add_argument("--cache-dir", default="data/cache/rollout_frames_v1")
    parser.add_argument("--smoke", action="store_true",
                        help="Run only 2 updates, use temporary output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Seeding ----
    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)
    rng_generator = torch.Generator(device="cpu").manual_seed(args.training_seed)

    # ---- Load world model ----
    anchor_path = Path(args.anchor)
    ac_path = Path(args.ac)
    if not anchor_path.is_file():
        parser.error(f"World-model anchor does not exist: {anchor_path}")
    if not ac_path.is_file():
        parser.error(f"Actor-Critic checkpoint does not exist: {ac_path}")
    anchor_hash = _hash_file(anchor_path)
    try:
        validate_ac_anchor_checkpoint(ac_path, anchor_hash)
    except ValueError as exc:
        parser.error(str(exc))

    # Create output only after cheap compatibility checks have passed.
    out = Path(args.out)
    if out.exists():
        parser.error(f"Output directory already exists: {out}")
    out.mkdir(parents=True)

    checkpoint = load_checkpoint(anchor_path, map_location=str(device))
    exp_config = checkpoint["config"]
    if exp_config is None:
        parser.error("Stage 6.1 requires a structured world-model checkpoint")
    model = model_from_checkpoint(
        checkpoint, action_dim=ACTION_DIM,
        tokenizer_eval_mode_override="mean",
    ).to(device)

    burn_in = model._temporal_config.sru_burn_in_steps
    seq_len = exp_config.data.sequence_len
    data_dir = Path(exp_config.data.dataset_dir)

    # ---- Load Actor-Critic without side effects ----
    joint_cfg = JointControllerCriticConfig()
    ac = load_ac_from_checkpoint(model, ac_path, device)
    ac_hash = _hash_file(ac_path)

    # ---- Data split (train only) ----
    val_ratio = getattr(getattr(exp_config, "data", None), "val_ratio", 0.2)
    train_files, val_files = collect_and_split(data_dir, args.data_split_seed, val_ratio)
    train_intersection = set(str(p) for p in train_files) & set(str(p) for p in val_files)
    if train_intersection:
        parser.error(f"Train/validation file overlap: {train_intersection}")

    dataset = RolloutDataset.from_file_list(
        train_files, sequence_len=seq_len,
        image_size=exp_config.data.image_size,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        recurrent_context=True,
        burn_in_steps=burn_in,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=6, pin_memory=True, persistent_workers=True,
        generator=rng_generator,
    )

    # ---- Trainer ----
    trainer = JointControllerCriticTrainer(model, ac, joint_cfg, device=device)
    updates = 2 if args.smoke else args.updates

    # ---- Trainable / frozen snapshots ----
    frozen_before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith("controller.")
    }
    sru_before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith("world_hd.")
    }
    perception_before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if any(name.startswith(p) for p in ["encoder.", "tokenizer.", "scorer.", "selector.", "spatial_hd."])
    }
    actor_before = {
        name: tensor.detach().cpu().clone() for name, tensor in ac.actor.state_dict().items()
    }
    target_critic_before = {
        name: tensor.detach().cpu().clone() for name, tensor in ac.target_critic.state_dict().items()
    }
    controller_before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.controller.state_dict().items()
    }
    critic_before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in ac.critic.state_dict().items()
    }

    # ---- Training loop ----
    iterator = iter(loader)
    rows = []
    start = time.time()

    for step in range(1, updates + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        metrics = trainer.train_step(batch)
        metrics["step"] = step
        rows.append(metrics)
        if step == 1 or step % 50 == 0 or step == updates:
            print(
                f"[{step:4d}/{updates}] total={metrics['total_loss']:.4f} "
                f"visible={metrics['visible_mse']:.4f} "
                f"masked={metrics['masked_mse']:.4f} "
                f"critic={metrics['critic_loss']:.4f}"
            )

    elapsed = time.time() - start

    # ---- Freeze boundary verification ----
    model_sd = {k: v.cpu() for k, v in model.state_dict().items()}
    actor_sd = {k: v.cpu() for k, v in ac.actor.state_dict().items()}
    target_sd = {k: v.cpu() for k, v in ac.target_critic.state_dict().items()}

    frozen_identity = all(
        torch.equal(tensor, model_sd[name])
        for name, tensor in frozen_before.items()
    )
    sru_identity = all(
        torch.equal(tensor, model_sd[name])
        for name, tensor in sru_before.items()
    )
    perception_identity = all(
        torch.equal(tensor, model_sd[name])
        for name, tensor in perception_before.items()
    )
    actor_identity = all(
        torch.equal(tensor, actor_sd[name])
        for name, tensor in actor_before.items()
    )
    target_critic_identity = all(
        torch.equal(tensor, target_sd[name])
        for name, tensor in target_critic_before.items()
    )
    controller_changed = any(
        not torch.equal(tensor, model.controller.state_dict()[name].detach().cpu())
        for name, tensor in controller_before.items()
    )
    critic_changed = any(
        not torch.equal(tensor, ac.critic.state_dict()[name].detach().cpu())
        for name, tensor in critic_before.items()
    )
    target_critic_polyak_updated = not target_critic_identity
    # Target Critic is updated via Polyak — that is expected.
    # All other frozen blocks must be bitwise unchanged.
    freeze_ok = all([
        frozen_identity, sru_identity, perception_identity,
        actor_identity,
    ])
    if not freeze_ok:
        import sys as _sys
        for name, tensor in frozen_before.items():
            after = model_sd[name]
            if not torch.equal(tensor, after):
                print(f"FROZEN PARAM CHANGED: {name}", file=_sys.stderr)
        for name, tensor in sru_before.items():
            after = model_sd[name]
            if not torch.equal(tensor, after):
                print(f"SRU CHANGED: {name}", file=_sys.stderr)
        for name, tensor in perception_before.items():
            after = model_sd[name]
            if not torch.equal(tensor, after):
                print(f"PERCEPTION CHANGED: {name}", file=_sys.stderr)
        for name, tensor in actor_before.items():
            after = actor_sd[name]
            if not torch.equal(tensor, after):
                print(f"ACTOR CHANGED: {name}", file=_sys.stderr)
        raise RuntimeError("Stage 6.1 freeze boundary was violated")
    if not controller_changed or not critic_changed or not target_critic_polyak_updated:
        raise RuntimeError(
            "Stage 6.1 expected update did not occur: "
            f"controller_changed={controller_changed}, "
            f"critic_changed={critic_changed}, "
            f"target_critic_polyak_updated={target_critic_polyak_updated}"
        )

    # ---- Save metrics ----
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # ---- Save checkpoints ----
    wm_path = save_checkpoint(
        out / "checkpoint_world_model.pt",
        model.state_dict(), exp_config,
        optimizer_state=trainer.optimizer.state_dict(),
        global_step=updates,
        metrics=rows[-1],
    )
    wm_hash = _hash_file(wm_path)
    ac_out = out / "checkpoint_actor_critic.pt"
    torch.save({
        "schema_version": 1,
        "kind": "imagined_actor_critic",
        "step": updates,
        "global_step": updates,
        "actor_critic": {
            "actor": ac.actor.state_dict(),
            "critic": ac.critic.state_dict(),
            "target_critic": ac.target_critic.state_dict(),
        },
        "optimizer": {
            "actor_optim": ac._actor_optim.state_dict(),
            "critic_optim": ac._critic_optim.state_dict(),
        },
        "config": dataclasses.asdict(joint_cfg),
        "actor_critic_config": dataclasses.asdict(ac.cfg),
        "anchor": {"path": str(wm_path.resolve()), "hash": wm_hash},
    }, ac_out)

    # ---- Provenance summary ----
    # Parameter counts
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    controller_params = sum(p.numel() for p in model.controller.parameters())
    critic_params = sum(p.numel() for p in ac.critic.parameters())
    actor_params = sum(p.numel() for p in ac.actor.parameters())
    sru_params = sum(p.numel() for p in model.world_hd.parameters())
    perception_total = sum(p.numel() for p in model.encoder.parameters())
    perception_total += sum(p.numel() for p in model.tokenizer.parameters())
    perception_total += sum(p.numel() for p in model.scorer.parameters())
    perception_total += sum(p.numel() for p in model.selector.parameters())
    perception_total += sum(p.numel() for p in model.spatial_hd.parameters())

    summary = {
        "provenance": {
            "anchor_path": str(anchor_path.resolve()),
            "anchor_hash": anchor_hash,
            "ac_path": str(ac_path.resolve()),
            "ac_hash": ac_hash,
            "data_root": str(data_dir.resolve()),
            "data_split_seed": args.data_split_seed,
            "training_seed": args.training_seed,
            "val_ratio": val_ratio,
            "train_files": sorted(str(p) for p in train_files),
            "val_files": sorted(str(p) for p in val_files),
            "n_train_files": len(train_files),
            "n_val_files": len(val_files),
            "train_val_disjoint": not train_intersection,
            "n_train_windows": len(dataset),
        },
        "freeze_boundary": {
            "controller_trunk": True,
            "online_critic": True,
            "controller_changed": controller_changed,
            "online_critic_changed": critic_changed,
            "frozen_world_model": frozen_identity,
            "sru_unchanged": sru_identity,
            "perception_unchanged": perception_identity,
            "actor_unchanged": actor_identity,
            "target_critic_polyak_updated": target_critic_polyak_updated,
            "controller_params": controller_params,
            "critic_params": critic_params,
            "target_critic_params": sum(p.numel() for p in ac.target_critic.parameters()),
            "actor_params (frozen)": actor_params,
            "sru_params (frozen)": sru_params,
            "perception_params (frozen)": perception_total,
            "total_frozen": frozen_params,
            "total_trainable": trainable_params,
        },
        "training": {
            "updates": updates,
            "batch_size": args.batch_size,
            "horizon": joint_cfg.horizon,
            "warmup_steps": joint_cfg.warmup_steps,
            "factual_weight": joint_cfg.factual_weight,
            "critic_weight": joint_cfg.critic_weight,
            "controller_lr": joint_cfg.controller_lr,
            "critic_lr": joint_cfg.critic_lr,
            "controller_grad_clip": joint_cfg.controller_grad_clip,
            "critic_grad_clip": joint_cfg.critic_grad_clip,
            "gamma": joint_cfg.gamma,
            "lambda_": joint_cfg.lambda_,
            "elapsed_s": elapsed,
        },
        "checkpoints": {
            "world_model": str(wm_path),
            "world_model_hash": wm_hash,
            "actor_critic": str(ac_out),
        },
        "loss_first": rows[0],
        "loss_last": rows[-1],
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
