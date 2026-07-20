"""Deterministic Actor evaluation in the real CarRacing environment.

Timing contract (approved)::

    obs_t = env state before action a_t
    belief z_t = Transformer(obs_t, action_{t-1})     — causal
    belief z_t = SRU(obs_t, action_{t-1})              — SRU
    Actor chooses a_t = mode(z_t)                      (deterministic)
    Reward(z_t, a_t) = r_hat_{t+1}                     (predicted)
    env.step(a_t) → obs_{t+1}, r_{t+1}                 (real)
"""

from __future__ import annotations

import csv
import hashlib
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from rwm.config.config import ACTION_DIM, VALUES_DIM
from rwm.envs.env import make_env
from rwm.evaluation.schema import SeedManifest
from rwm.models.actor_critic import ActorCritic
from rwm.models.rwm.model import ReducedWorldModel
from rwm.trainers.imagined_actor_critic import ImaginedACTrainer
from rwm.utils.preprocess_observation import preprocess_obs


_MAX_STEPS = 1000
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def verify_actor_checkpoint_anchor(
    anchor_path: Path,
    ac_checkpoint_path: Path,
) -> Tuple[str, bool]:
    """Verify that an Actor-Critic checkpoint belongs to ``anchor_path``.

    Returns the current anchor hash and whether the Actor-Critic checkpoint
    contained verifiable provenance. Legacy checkpoints without an embedded
    anchor hash remain loadable, but emit a warning instead of claiming
    successful verification.
    """
    digest = hashlib.sha256()
    digest.update(anchor_path.read_bytes())
    actual_hash = digest.hexdigest()[:16]

    checkpoint = torch.load(
        ac_checkpoint_path, map_location="cpu", weights_only=False,
    )
    embedded_anchor = checkpoint.get("anchor") or {}
    embedded_hash = embedded_anchor.get("hash")

    if embedded_hash is None:
        warnings.warn(
            "Actor-Critic checkpoint has no embedded anchor hash; "
            "anchor integrity cannot be verified",
            RuntimeWarning,
            stacklevel=2,
        )
        return actual_hash, False

    if embedded_hash != actual_hash:
        raise ValueError(
            f"Anchor hash mismatch: checkpoint expects {embedded_hash}, "
            f"got {actual_hash} from {anchor_path}"
        )

    return actual_hash, True


def _validate_seed(seed: int, manifest: SeedManifest) -> None:
    """Allow real-policy evaluation only on the locked manifest's dev split."""
    manifest.assert_valid()
    s = str(seed)
    if s not in manifest.entries:
        raise ValueError(
            f"seed {seed} not in manifest; available dev seeds: "
            f"{list(manifest.entries.keys())}"
        )
    if manifest.entries[s] not in ("dev",):
        raise ValueError(
            f"seed {seed} is split '{manifest.entries[s]}', "
            f"only dev seeds are allowed for evaluation"
        )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

class EpisodeResult:
    """Record of one evaluated episode."""

    def __init__(self, seed: int, rng_seed: Optional[int] = None) -> None:
        self.seed = seed
        self.rng_seed = rng_seed
        self.steps: List[Dict[str, float]] = []
        self.terminated: bool = False
        self.truncated: bool = False
        self.cumulative_reward: float = 0.0

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def record_step(
        self,
        action: np.ndarray,
        reward_true: float,
        reward_pred: float,
        value: float,
        logstd: np.ndarray,
        patches: Optional[np.ndarray] = None,
    ) -> None:
        self.steps.append({
            "action_steer": float(action[0]),
            "action_gas": float(action[1]),
            "action_brake": float(action[2]),
            "reward_true": float(reward_true),
            "reward_pred": float(reward_pred),
            "value": float(value),
            "logstd_steer": float(logstd[0]),
            "logstd_gas": float(logstd[1]),
            "logstd_brake": float(logstd[2]),
            "patches": patches.tolist() if patches is not None else [],
        })
        self.cumulative_reward += reward_true

    def set_done(self, terminated: bool, truncated: bool) -> None:
        self.terminated = terminated
        self.truncated = truncated


def run_episode(
    model: ReducedWorldModel,
    ac: ActorCritic,
    seed: int,
    manifest: SeedManifest,
    max_steps: int = _MAX_STEPS,
    render_mode: str = "rgb_array",
) -> EpisodeResult:
    """Run one deterministic evaluation episode.

    Supports both causal Transformer and MinimalSRU backends.
    Causal carries ``history``/``lengths``; SRU carries ``temporal_state``.

    Parameters
    ----------
    model:
        Frozen world model.
    ac:
        Actor-Critic module in eval mode.
    seed:
        Environment seed (must be a dev seed).
    max_steps:
        Maximum steps per episode.
    render_mode:
        Gymnasium render mode.

    Returns
    -------
    ``EpisodeResult`` with per-step records.
    """
    _validate_seed(seed, manifest)
    model.eval()
    ac.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    is_sru = getattr(model, "_temporal_backend", "causal_transformer") == "minimal_sru"

    env = make_env("car_racing", render_mode=render_mode)
    obs, _ = env.reset(seed=int(seed))
    result = EpisodeResult(seed)

    prev_action_np = np.zeros(ACTION_DIM, dtype=np.float32)
    history_t = None
    lengths_t = None
    temporal_state_t = None  # SRU only

    for step in range(max_steps):
        # Preprocess observation.
        obs_t = preprocess_obs(obs).to(next(model.parameters()).device)

        prev_action_t = torch.from_numpy(prev_action_np).float().unsqueeze(0).to(obs_t.device)

        with torch.no_grad():
            out = model(
                img=obs_t,
                prev_action=prev_action_t,
                current_action=prev_action_t,  # placeholder, will re-run
                history=history_t,
                lengths=lengths_t,
                force_keep_input=True,
                temporal_state=temporal_state_t,
            )

            z_t = out.world_state
            c_t = ac.encode(z_t)
            dist = ac.actor(c_t)
            a_t = dist.mode()  # (1, A)
            logstd_t = dist.logstd  # (1, A)
            value_t = ac.critic(c_t)  # (1, 1)
            # Predict reward with the actual chosen action.
            _, r_hat_t = model.controller(z_t, a_t)

        # Record.
        action_np = a_t.squeeze(0).detach().cpu().numpy()
        logstd_np = logstd_t.squeeze(0).detach().cpu().numpy()
        r_hat_val = r_hat_t.detach().item()
        value_val = value_t.detach().item()

        # Execute in environment.
        real_action = np.clip(action_np, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        next_obs, reward_real, terminated, truncated, _ = env.step(real_action)

        result.record_step(
            action=action_np,
            reward_true=reward_real,
            reward_pred=r_hat_val,
            value=value_val,
            logstd=logstd_np,
            patches=out.indices.squeeze(0).cpu().numpy() if out.indices is not None else None,
        )

        if terminated or truncated:
            result.set_done(terminated, truncated)
            break

        # Advance state.
        prev_action_np = action_np
        if is_sru:
            temporal_state_t = out.temporal_state
        else:
            history_t = out.history
            lengths_t = out.lengths
        obs = next_obs

    env.close()
    return result


# ---------------------------------------------------------------------------
# Zero-action baseline
# ---------------------------------------------------------------------------

def run_zero_baseline(
    seed: int,
    manifest: SeedManifest,
    max_steps: int = _MAX_STEPS,
    render_mode: str = "rgb_array",
) -> EpisodeResult:
    """Run one episode with zero actions (do-nothing baseline)."""
    _validate_seed(seed, manifest)
    env = make_env("car_racing", render_mode=render_mode)
    obs, _ = env.reset(seed=int(seed))
    result = EpisodeResult(seed)
    zero_action = np.zeros(ACTION_DIM, dtype=np.float32)

    for step in range(max_steps):
        _, reward_real, terminated, truncated, _ = env.step(zero_action)
        result.record_step(
            action=zero_action,
            reward_true=reward_real,
            reward_pred=0.0,
            value=0.0,
            logstd=np.zeros(ACTION_DIM),
        )
        if terminated or truncated:
            result.set_done(terminated, truncated)
            break

    env.close()
    return result


def run_random_baseline(
    seed: int,
    manifest: SeedManifest,
    max_steps: int = _MAX_STEPS,
    render_mode: str = "rgb_array",
    rng_seed: Optional[int] = None,
) -> EpisodeResult:
    """Run one episode with deterministic random actions (bounded).

    Actions are sampled from a seeded RNG at each step:
      steer  ~ Uniform(-1, 1)
      gas    ~ Uniform(0, 1)
      brake  ~ Uniform(0, 1)

    Parameters
    ----------
    rng_seed:
        If given, the random policy is deterministic across calls.
        Defaults to ``seed + 10000``.
    """
    _validate_seed(seed, manifest)
    if rng_seed is None:
        rng_seed = seed + 10000
    rng = np.random.RandomState(rng_seed)

    env = make_env("car_racing", render_mode=render_mode)
    obs, _ = env.reset(seed=int(seed))
    result = EpisodeResult(seed, rng_seed=rng_seed)

    for step in range(max_steps):
        steer = float(rng.uniform(-1.0, 1.0))
        gas = float(rng.uniform(0.0, 1.0))
        brake = float(rng.uniform(0.0, 1.0))
        action_np = np.array([steer, gas, brake], dtype=np.float32)

        _, reward_real, terminated, truncated, _ = env.step(action_np)
        result.record_step(
            action=action_np,
            reward_true=reward_real,
            reward_pred=0.0,
            value=0.0,
            logstd=np.zeros(ACTION_DIM),
        )
        if terminated or truncated:
            result.set_done(terminated, truncated)
            break

    env.close()
    return result


# ---------------------------------------------------------------------------
# CSV / JSON persistence
# ---------------------------------------------------------------------------

def save_episode_csv(episode: EpisodeResult, path: Path) -> None:
    """Save per-step data as CSV."""
    fieldnames = [
        "seed", "step",
        "action_steer", "action_gas", "action_brake",
        "reward_true", "reward_pred", "value",
        "logstd_steer", "logstd_gas", "logstd_brake",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, s in enumerate(episode.steps):
            row = {
                "seed": episode.seed,
                "step": i,
                **{k: s[k] for k in fieldnames if k in s},
            }
            writer.writerow(row)


def save_episode_json(episode: EpisodeResult, path: Path) -> None:
    """Save episode summary and per-step data as JSON."""
    data: Dict = {
        "seed": episode.seed,
        "n_steps": episode.n_steps,
        "cumulative_reward": episode.cumulative_reward,
        "terminated": episode.terminated,
        "truncated": episode.truncated,
    }
    if episode.rng_seed is not None:
        data["rng_seed"] = episode.rng_seed
    data["steps"] = episode.steps

    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def compute_reward_mse_mae(episode: EpisodeResult) -> Tuple[float, float]:
    """Compute MSE and MAE between predicted and true rewards."""
    if not episode.steps:
        return 0.0, 0.0
    preds = np.array([s["reward_pred"] for s in episode.steps])
    trues = np.array([s["reward_true"] for s in episode.steps])
    mse = float(np.mean((preds - trues) ** 2))
    mae = float(np.mean(np.abs(preds - trues)))
    return mse, mae


def mean_action(episode: EpisodeResult) -> np.ndarray:
    if not episode.steps:
        return np.zeros(ACTION_DIM)
    acts = np.array([[s["action_steer"], s["action_gas"], s["action_brake"]]
                     for s in episode.steps])
    return acts.mean(axis=0)
