#!/usr/bin/env python3
"""Evaluate a trained Actor-Critic policy in the real CarRacing environment.

Usage::

    # Evaluate trained checkpoint
    python scripts/evaluate_real_env.py \\
        --checkpoint runs/imagined_actor_critic/stage5_3_train/checkpoints/ac_checkpoint_2000.pt \\
        --anchor runs/component_refinement/08_masked_reward_anchor/seed42/checkpoint_best.pt \\
        --out runs/imagined_actor_critic/stage5_3_eval

    # Zero-action baseline
    python scripts/evaluate_real_env.py --baseline --out runs/imagined_actor_critic/stage5_3_baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from rwm.config.config import ACTION_DIM
from rwm.evaluation.plotting import (
    plot_action_traces,
    plot_cumulative_rewards,
    plot_patch_overlay_samples,
    plot_reward_comparison,
)
from rwm.evaluation.real_env_evaluator import (
    compute_reward_mse_mae,
    mean_action,
    run_episode,
    run_zero_baseline,
    save_episode_csv,
    save_episode_json,
)
from rwm.evaluation.schema import Split, load_seed_manifest
from rwm.trainers.imagined_actor_critic import ImaginedACTrainer
from rwm.utils.checkpointing import load_checkpoint, model_from_checkpoint


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _load_model_and_ac(anchor_path: Path, ac_checkpoint_path: Path, device):
    """Load frozen world model and trained Actor-Critic."""
    ckpt = load_checkpoint(anchor_path)
    model = model_from_checkpoint(ckpt, action_dim=ACTION_DIM,
                                   tokenizer_eval_mode_override="mean")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)

    # Build a minimal trainer just for AC loading.
    from torch.utils.data import DataLoader
    from rwm.data.rollout_dataset import RolloutDataset
    data_root = Path(
        ckpt["config"].data.dataset_dir if ckpt["config"]
        else "data/rollouts/rwm_deterministic/scenario_0"
    )
    dummy_ds = RolloutDataset(
        root_dir=data_root, sequence_len=16, image_size=64,
    )
    dummy_loader = DataLoader(dummy_ds, batch_size=1, shuffle=False, drop_last=True)
    from rwm.trainers.imagined_actor_critic import ImaginedACTrainingConfig
    trainer = ImaginedACTrainer(
        model=model, train_loader=dummy_loader,
        train_cfg=ImaginedACTrainingConfig(max_batches=1),
        device=device, out_dir=Path("/tmp/_ac_loader"),
    )
    trainer.load_actor_critic_checkpoint(ac_checkpoint_path)
    return model, trainer.ac


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen Actor-Critic policy in real CarRacing",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="",
        help="Path to Actor-Critic checkpoint (.pt)",
    )
    parser.add_argument(
        "--anchor", type=str, default="",
        help="Path to frozen world-model anchor (.pt)",
    )
    parser.add_argument(
        "--out", type=str, default="runs/real_env_eval",
        help="Output directory",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Run zero-action baseline instead of loading a policy",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Dev seeds to evaluate (default: first three from manifest)",
    )
    parser.add_argument(
        "--seed-manifest", type=str, default="data/eval/seeds.json",
        help="Locked seed manifest; only its dev seeds are accepted.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render environment (requires display)",
    )
    args = parser.parse_args()

    if not args.baseline and (not args.checkpoint or not args.anchor):
        parser.error("--checkpoint and --anchor are required unless --baseline is set")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render_mode = "human" if args.render else "rgb_array"

    manifest_path = Path(args.seed_manifest)
    if not manifest_path.is_file():
        parser.error(f"seed manifest not found: {manifest_path}")
    manifest = load_seed_manifest(manifest_path)
    manifest.assert_valid()
    dev_seeds = sorted(
        int(seed) for seed, split in manifest.entries.items()
        if split == Split.DEV.value
    )
    seeds = args.seeds if args.seeds is not None else dev_seeds[:3]
    if not seeds:
        parser.error("seed manifest contains no dev seeds")

    if args.baseline:
        print("Running zero-action baseline...")
        results = {}
        policy_name = "zero_action"
        for seed in seeds:
            print(f"  Seed {seed}...")
            ep = run_zero_baseline(seed, max_steps=args.max_steps,
                                    manifest=manifest, render_mode=render_mode)
            results[seed] = ep
            csv_path = out_dir / f"episode_seed{seed}.csv"
            json_path = out_dir / f"episode_seed{seed}.json"
            save_episode_csv(ep, csv_path)
            save_episode_json(ep, json_path)
            print(f"    Steps: {ep.n_steps}, Reward: {ep.cumulative_reward:.2f}")
        # Save aggregate.
        summary = {str(s): {"cumulative_reward": ep.cumulative_reward,
                            "n_steps": ep.n_steps,
                            "terminated": ep.terminated,
                            "truncated": ep.truncated}
                   for s, ep in results.items()}
        summary["seed_manifest_path"] = str(manifest_path.resolve())
        summary["seed_manifest_hash"] = _file_hash(manifest_path)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"\nDone. Output in {out_dir}")
        return

    # Load model and AC.
    anchor_path = Path(args.anchor)
    ckpt_path = Path(args.checkpoint)
    anchor_hash = _file_hash(anchor_path)
    print(f"Anchor: {anchor_path} (hash: {anchor_hash})")
    print(f"Checkpoint: {ckpt_path}")

    model, ac = _load_model_and_ac(anchor_path, ckpt_path, device)
    print(f"Model and Actor-Critic loaded on {device}")

    # Verify anchor hash.
    wm_hash_now = _file_hash(anchor_path)
    assert wm_hash_now == anchor_hash, (
        f"Anchor hash mismatch: {wm_hash_now} != {anchor_hash}"
    )
    print(f"Anchor hash verified: {anchor_hash}")

    # Run evaluation.
    results: dict = {}
    for seed in seeds:
        print(f"\nEvaluating seed {seed}...")
        ep = run_episode(model, ac, seed, max_steps=args.max_steps,
                          manifest=manifest, render_mode=render_mode)
        results[seed] = ep

        csv_path = out_dir / f"episode_seed{seed}.csv"
        json_path = out_dir / f"episode_seed{seed}.json"
        save_episode_csv(ep, csv_path)
        save_episode_json(ep, json_path)

        mse, mae = compute_reward_mse_mae(ep)
        mean_act = mean_action(ep)
        print(f"  Steps: {ep.n_steps}, Reward: {ep.cumulative_reward:.2f}, "
              f"Term: {ep.terminated}, Trunc: {ep.truncated}")
        print(f"  Reward MSE: {mse:.4f}, MAE: {mae:.4f}")
        print(f"  Mean action: steer={mean_act[0]:.3f}, "
              f"gas={mean_act[1]:.3f}, brake={mean_act[2]:.3f}")

    # Save summary.
    summary = {str(s): {"cumulative_reward": ep.cumulative_reward,
                        "n_steps": ep.n_steps,
                        "terminated": ep.terminated,
                        "truncated": ep.truncated,
                        "reward_mse": compute_reward_mse_mae(ep)[0],
                        "reward_mae": compute_reward_mse_mae(ep)[1],
                        "mean_action": mean_action(ep).tolist()}
               for s, ep in results.items()}
    summary["anchor_path"] = str(anchor_path.resolve())
    summary["anchor_hash"] = anchor_hash
    summary["checkpoint_path"] = str(ckpt_path.resolve())
    summary["seed_manifest_path"] = str(manifest_path.resolve())
    summary["seed_manifest_hash"] = _file_hash(manifest_path)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # Plots.
    first_seed = seeds[0]
    ep0 = results[first_seed]
    plot_cumulative_rewards(
        {"actor": list(results.values())},
        out_dir / "cumulative_rewards.png",
    )
    plot_reward_comparison(ep0, out_dir / "reward_comparison.png")
    plot_action_traces(ep0, out_dir / "action_traces.png")
    plot_patch_overlay_samples(ep0, out_dir / "patch_overlay.png",
                                step_interval=50, n_samples=4)

    print(f"\nPlots saved to {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
