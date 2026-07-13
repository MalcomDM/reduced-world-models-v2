"""Interactive human reward-prediction evaluation for the current world model."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import typer
from PIL import Image

from rwm.config.config import ACTION_DIM, INPUT_DIM
from rwm.models.rwm.model import ReducedWorldModel
from rwm.policies.human_policy import HumanPolicy
from rwm.utils.checkpointing import load_checkpoint


def preprocess_obs(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a CarRacing RGB frame to the model's ``(1, 3, 64, 64)`` input."""
    image = Image.fromarray(obs).resize(INPUT_DIM[:2])
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def load_model(checkpoint_path: Path, device: torch.device) -> ReducedWorldModel:
    """Load a structured Stage-2 checkpoint into the active architecture."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))
    model = ReducedWorldModel(action_dim=ACTION_DIM)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def _save_reward_log(path: Path, rows: list[tuple[int, int, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "step", "reward_true", "reward_pred"])
        writer.writerows(rows)


def run(
    ckpt: Path = typer.Argument(..., exists=True, help="Structured world-model checkpoint (.pt)"),
    env_name: str = typer.Option("CarRacing-v3", help="Gymnasium environment name"),
    log_path: Path = typer.Option(Path("runs/manual_reward_eval.csv"), help="CSV output path"),
    fps: int = typer.Option(60, min=1, help="Maximum interactive loop rate"),
    max_steps: Optional[int] = typer.Option(None, min=1, help="Stop after this many total steps"),
) -> None:
    """Drive CarRacing and plot predicted versus immediate real reward live.

    At each step the human selects ``action[t]`` from ``obs[t]``.  The model
    receives ``obs[t]``, ``action[t-1]`` and ``action[t]``, predicting the
    reward returned by that same subsequent ``env.step(action[t])``.
    """
    import matplotlib.pyplot as plt
    import pygame

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt, device)
    env = gym.make(env_name, render_mode="human", continuous=True)
    # CarRacing creates its SDL video context lazily on reset.  Do this before
    # the keyboard policy touches pygame's event queue.
    obs, _ = env.reset()
    policy = HumanPolicy()
    policy.reset()
    clock = pygame.time.Clock()

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set(xlabel="Step within episode", ylabel="Immediate reward", title="Real vs predicted reward")
    (line_true,) = ax.plot([], [], label="Real", color="tab:blue")
    (line_pred,) = ax.plot([], [], label="Predicted", color="tab:orange")
    ax.legend()
    plt.show(block=False)

    prev_action = torch.zeros(1, ACTION_DIM, device=device)
    history: Optional[torch.Tensor] = None
    lengths: Optional[torch.Tensor] = None
    true_rewards: list[float] = []
    predicted_rewards: list[float] = []
    rows: list[tuple[int, int, float, float]] = []
    episode, step, total_steps = 0, 0, 0

    try:
        while policy.is_running() and (max_steps is None or total_steps < max_steps):
            # The current action is deliberately chosen before prediction: it
            # conditions r[t] while only a[t-1] enters the causal belief.
            action = policy.act(obs)
            current_action = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            frame = preprocess_obs(obs, device)

            with torch.no_grad():
                output = model(
                    img=frame,
                    prev_action=prev_action,
                    current_action=current_action,
                    history=history,
                    lengths=lengths,
                )
            prediction = float(output.reward_pred.squeeze().item())
            next_obs, reward, terminated, truncated, _ = env.step(action)

            true_rewards.append(float(reward))
            predicted_rewards.append(prediction)
            rows.append((episode, step, float(reward), prediction))
            xs = np.arange(len(true_rewards))
            line_true.set_data(xs, true_rewards)
            line_pred.set_data(xs, predicted_rewards)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            history, lengths = output.history, output.lengths
            prev_action = current_action
            obs = next_obs
            step += 1
            total_steps += 1

            if terminated or truncated:
                obs, _ = env.reset()
                prev_action = torch.zeros(1, ACTION_DIM, device=device)
                history, lengths = None, None
                true_rewards.clear()
                predicted_rewards.clear()
                episode += 1
                step = 0

            clock.tick(fps)
    finally:
        _save_reward_log(log_path, rows)
        env.close()
        pygame.quit()
        plt.ioff()
        plt.close(fig)

    typer.echo(f"Saved {len(rows)} aligned reward pairs to {log_path}")
