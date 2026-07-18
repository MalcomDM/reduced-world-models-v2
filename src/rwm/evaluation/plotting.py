"""Compact evaluation plots for Stage 5.3 real-environment evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rwm.evaluation.real_env_evaluator import EpisodeResult


def plot_cumulative_rewards(
    results: Dict[str, List[EpisodeResult]],
    save_path: Path,
) -> None:
    """Bar chart of cumulative real reward per seed per policy."""
    policies = list(results.keys())
    seeds = sorted({r.seed for rs in results.values() for r in rs})

    x = np.arange(len(seeds))
    width = 0.8 / len(policies)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, policy in enumerate(policies):
        vals = [
            next((r.cumulative_reward for r in results[policy] if r.seed == s), 0)
            for s in seeds
        ]
        offset = (i - len(policies) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=policy)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("Cumulative real reward")
    ax.set_title("Cumulative real reward per seed")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_reward_comparison(
    episode: EpisodeResult,
    save_path: Path,
) -> None:
    """Real vs predicted reward for a single episode."""
    steps = np.arange(episode.n_steps)
    trues = np.array([s["reward_true"] for s in episode.steps])
    preds = np.array([s["reward_pred"] for s in episode.steps])

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(steps, trues, label="True reward", alpha=0.8)
    ax.plot(steps, preds, label="Predicted reward", alpha=0.8, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title(f"Real vs predicted reward (seed {episode.seed})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_action_traces(
    episode: EpisodeResult,
    save_path: Path,
) -> None:
    """Action traces for a single episode."""
    steps = np.arange(episode.n_steps)
    steer = np.array([s["action_steer"] for s in episode.steps])
    gas = np.array([s["action_gas"] for s in episode.steps])
    brake = np.array([s["action_brake"] for s in episode.steps])

    fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(steps, steer, label="Steer", color="C0")
    axes[0].axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    axes[0].axhline(-1.0, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("Steer")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(-1.1, 1.1)

    axes[1].plot(steps, gas, label="Gas", color="C1")
    axes[1].axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_ylabel("Gas")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(-0.05, 1.05)

    axes[2].plot(steps, brake, label="Brake", color="C2")
    axes[2].axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    axes[2].set_ylabel("Brake")
    axes[2].set_xlabel("Step")
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(-0.05, 1.05)

    fig.suptitle(f"Action traces (seed {episode.seed})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_patch_overlay_samples(
    episode: EpisodeResult,
    save_path,
    step_interval: int = 50,
    n_samples: int = 4,
) -> None:
    """Plot selected-patch indices every `step_interval` steps.

    Since the evaluator doesn't store observation pixels, this plots the
    selected patch indices per step as a heatmap-like scatter.
    """
    steps_with_patches = [
        (i, s) for i, s in enumerate(episode.steps)
        if s.get("patches") and len(s["patches"]) > 0
    ]
    if not steps_with_patches:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No patch data", ha="center", va="center")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return

    sampled = steps_with_patches[::step_interval][:n_samples]
    fig, axes = plt.subplots(1, len(sampled), figsize=(4 * len(sampled), 3),
                              squeeze=False)
    for ax, (step_idx, s) in zip(axes[0], sampled):
        patches = np.array(s["patches"])
        ax.scatter(patches % 15, patches // 15, s=10, alpha=0.7)
        ax.set_xlim(-1, 15)
        ax.set_ylim(-1, 15)
        ax.set_aspect("equal")
        ax.set_title(f"Step {step_idx}")
        ax.set_xlabel("Patch x")
        ax.set_ylabel("Patch y")
    fig.suptitle(f"Selected patch indices (seed {episode.seed})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
