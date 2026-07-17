"""Full-episode evaluator that processes each transition once.

Does not use overlapping windows.  Consumes declared evaluation manifests
and never trains.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from rwm.evaluation.schema import (
    EpisodeMetadata,
    SeedManifest,
    load_episode_metadata,
    load_seed_manifest,
    Split,
    validate_episode_integrity,
)


@dataclass
class EpisodeResult:
    """Metrics for one evaluation episode."""
    episode_id: str
    split: str
    seed: int
    steps: int

    # Natural metrics
    mse: float = 0.0
    mae: float = 0.0
    cum_reward_true: float = 0.0
    cum_reward_pred: float = 0.0

    # Reward-event metrics
    reward_positive_frac: float = 0.0
    reward_neutral_frac: float = 0.0

    # Stratified
    startup_mse: float = 0.0
    steady_mse: float = 0.0

    # Human-readable reward slices.  Each has transition count and observed /
    # predicted mean reward, allowing a benchmark notebook to inspect where an
    # error comes from without replaying the model.
    startup_n: int = 0
    startup_mean_true: float = 0.0
    startup_mean_pred: float = 0.0
    steady_n: int = 0
    steady_mean_true: float = 0.0
    steady_mean_pred: float = 0.0
    reward_event_steady_n: int = 0
    reward_event_steady_mean_true: float = 0.0
    reward_event_steady_mean_pred: float = 0.0
    baseline_steady_n: int = 0
    baseline_steady_mean_true: float = 0.0
    baseline_steady_mean_pred: float = 0.0

    # Baseline
    baseline_mse: float = 0.0


@dataclass
class EvaluationSummary:
    """Aggregated metrics across a split."""
    split: str
    num_episodes: int
    total_steps: int
    avg_mse: float
    avg_mae: float
    avg_baseline_mse: float
    avg_cum_reward_true: float
    avg_cum_reward_pred: float
    reward_positive_frac: float
    reward_neutral_frac: float
    per_episode: List[EpisodeResult]


def load_evaluation_episode(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, EpisodeMetadata]:
    """Load one evaluation episode ``.npz`` and its ``.episode.json`` sidecar.

    Returns (obs, actions, rewards, metadata).
    """
    data = np.load(path)
    meta_path = path.with_suffix(".episode.json")
    meta = load_episode_metadata(meta_path)

    obs = data["obs"]
    actions = data["action"]
    rewards = data["reward"]

    return obs, actions, rewards, meta


def assert_evaluation_episode_integrity(
    path: Path,
    manifest_path: Path,
    expected_split: Optional[str] = None,
) -> EpisodeMetadata:
    """Load an episode and reject it if its locked provenance is invalid."""
    meta = load_episode_metadata(path.with_suffix(".episode.json"))
    manifest = load_seed_manifest(manifest_path)
    issues = validate_episode_integrity(meta, manifest, manifest_path, expected_split)
    if issues:
        raise ValueError(f"Invalid evaluation episode {path}: " + "; ".join(issues))
    return meta


def evaluate_episode(
    model: torch.nn.Module,
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    meta: EpisodeMetadata,
    train_reward_mean: float = 0.0,
    sequence_len: int = 16,
    startup_threshold: int = 50,
    device: str = "cpu",
) -> EpisodeResult:
    """Evaluate a model on one full episode, processing each transition once.

    Parameters
    ----------
    model:
        ``ReducedWorldModel`` in eval mode.
    obs, actions, rewards:
        Episode data (T, H, W, C), (T, A), (T,).
    meta:
        Episode metadata.
    train_reward_mean:
        Mean reward from training set (for baseline MSE).
    sequence_len:
        Context length for the Transformer history buffer.

    Returns
    -------
    ``EpisodeResult`` with per-episode metrics.
    """
    from rwm.types import WorldModelOutput
    from rwm.utils.history_buffer import HistoryBuffer
    from rwm.config.config import VALUES_DIM, ACTION_DIM, SEQ_LEN

    model.eval()
    T = len(obs)

    preds: List[float] = []
    history: Optional[torch.Tensor] = None
    lengths: Optional[torch.Tensor] = None
    prev_action = torch.zeros(1, 3, device=device)

    with torch.no_grad():
        for t in range(T):
            # Preprocess observation
            from PIL import Image
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((64, 64), antialias=True),
            ])
            img = transform(Image.fromarray(obs[t])).unsqueeze(0).to(device)

            current_action = torch.from_numpy(actions[t:t+1]).float().to(device)

            out: WorldModelOutput = model(
                img=img,
                prev_action=prev_action,
                current_action=current_action,
                history=history,
                lengths=lengths,
                force_keep_input=True,
            )
            preds.append(float(out.reward_pred.squeeze(-1).item()))
            history = out.history
            lengths = out.lengths
            prev_action = current_action

    preds_arr = np.array(preds, dtype=np.float32)
    true_arr = rewards[:len(preds)].astype(np.float32)
    n = len(preds_arr)

    mse = float(np.mean((preds_arr - true_arr) ** 2))
    mae = float(np.mean(np.abs(preds_arr - true_arr)))
    baseline_mse = float(np.mean((true_arr - train_reward_mean) ** 2))

    positive_mask = true_arr > -0.1
    neutral_mask = ~positive_mask
    reward_positive_frac = float(positive_mask.mean()) if n > 0 else 0.0
    reward_neutral_frac = float(neutral_mask.mean()) if n > 0 else 0.0

    def _slice_summary(mask: np.ndarray) -> tuple[int, float, float]:
        if not mask.any():
            return 0, 0.0, 0.0
        return int(mask.sum()), float(true_arr[mask].mean()), float(preds_arr[mask].mean())

    startup_mask = np.arange(n) < startup_threshold
    startup_mse = float(np.mean((preds_arr[startup_mask] - true_arr[startup_mask]) ** 2)) if startup_mask.any() else 0.0
    steady_mask = np.arange(n) >= startup_threshold
    steady_mse = float(np.mean((preds_arr[steady_mask] - true_arr[steady_mask]) ** 2)) if steady_mask.any() else 0.0
    reward_event_steady_mask = steady_mask & positive_mask
    baseline_steady_mask = steady_mask & np.isclose(true_arr, -0.1)
    startup_n, startup_mean_true, startup_mean_pred = _slice_summary(startup_mask)
    steady_n, steady_mean_true, steady_mean_pred = _slice_summary(steady_mask)
    reward_event_steady_n, reward_event_steady_mean_true, reward_event_steady_mean_pred = _slice_summary(reward_event_steady_mask)
    baseline_steady_n, baseline_steady_mean_true, baseline_steady_mean_pred = _slice_summary(baseline_steady_mask)

    return EpisodeResult(
        episode_id=meta.episode_id,
        split=meta.split,
        seed=meta.track_seed,
        steps=n,
        mse=mse,
        mae=mae,
        cum_reward_true=float(true_arr.sum()),
        cum_reward_pred=float(preds_arr.sum()),
        reward_positive_frac=reward_positive_frac,
        reward_neutral_frac=reward_neutral_frac,
        startup_mse=startup_mse,
        steady_mse=steady_mse,
        startup_n=startup_n,
        startup_mean_true=startup_mean_true,
        startup_mean_pred=startup_mean_pred,
        steady_n=steady_n,
        steady_mean_true=steady_mean_true,
        steady_mean_pred=steady_mean_pred,
        reward_event_steady_n=reward_event_steady_n,
        reward_event_steady_mean_true=reward_event_steady_mean_true,
        reward_event_steady_mean_pred=reward_event_steady_mean_pred,
        baseline_steady_n=baseline_steady_n,
        baseline_steady_mean_true=baseline_steady_mean_true,
        baseline_steady_mean_pred=baseline_steady_mean_pred,
        baseline_mse=baseline_mse,
    )


def evaluate_split(
    model: torch.nn.Module,
    split_dir: Path,
    split_name: str,
    manifest_path: Path,
    train_reward_mean: float = 0.0,
    device: str = "cpu",
) -> EvaluationSummary:
    """Evaluate a model on all episodes in a split directory.

    ``split_dir`` should contain ``.npz`` files with matching ``.episode.json``
    sidecars.  A manifest is required: metric reporting refuses tampered,
    wrong-split, or non-evaluation artifacts.
    """
    npz_files = sorted(split_dir.glob("*.npz"))
    results: List[EpisodeResult] = []

    for f in npz_files:
        obs, acts, rew, meta = load_evaluation_episode(f)
        # Validate before model execution so invalid data cannot contribute to
        # thesis metrics even if a caller bypasses the CLI status command.
        assert_evaluation_episode_integrity(f, manifest_path, split_name)
        res = evaluate_episode(
            model, obs, acts, rew, meta,
            train_reward_mean=train_reward_mean,
            device=device,
        )
        results.append(res)

    n_ep = len(results)
    total_steps = sum(r.steps for r in results)
    avg_mse = float(np.mean([r.mse for r in results])) if results else 0.0
    avg_mae = float(np.mean([r.mae for r in results])) if results else 0.0
    avg_baseline = float(np.mean([r.baseline_mse for r in results])) if results else 0.0

    return EvaluationSummary(
        split=split_name,
        num_episodes=n_ep,
        total_steps=total_steps,
        avg_mse=avg_mse,
        avg_mae=avg_mae,
        avg_baseline_mse=avg_baseline,
        avg_cum_reward_true=float(np.mean([r.cum_reward_true for r in results])) if results else 0.0,
        avg_cum_reward_pred=float(np.mean([r.cum_reward_pred for r in results])) if results else 0.0,
        reward_positive_frac=float(np.mean([r.reward_positive_frac for r in results])) if results else 0.0,
        reward_neutral_frac=float(np.mean([r.reward_neutral_frac for r in results])) if results else 0.0,
        per_episode=results,
    )
