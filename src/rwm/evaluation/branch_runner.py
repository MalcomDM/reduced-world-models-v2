"""Deterministic branch runner.

Resets a CarRacing environment at a fixed seed, replays a supplied action
prefix to reproduce the same state, then executes named action branches.

Results are saved under ``data/eval/branches/``, never under split dirs,
so they are not mistaken for labeled evaluation episodes.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium import Env

from rwm.envs.env import make_env


_BRANCH_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BranchMetadata:
    """Sidecar metadata for a branch experiment."""
    schema_version: int = _BRANCH_SCHEMA_VERSION
    purpose: str = "evaluation_only_branch"
    seed: int = 0
    source_manifest_hash: str = ""
    prefix_provenance: str = ""
    branch_definitions: Dict[str, Tuple[int, ...]] = dataclasses.field(default_factory=dict)
    env_id: str = "CarRacing-v3"
    env_version: str = ""
    created_at: str = ""
    git_commit: str = ""
    config_ref: str = ""


@dataclass(frozen=True)
class BranchResult:
    """Result of executing one action branch from a fixed anchor state."""
    branch_name: str
    actions: np.ndarray          # (T_branch, A)
    observations: np.ndarray     # (T_branch, H, W, C)
    rewards: np.ndarray          # (T_branch,)
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class BranchExperiment:
    """Full branch experiment for one seed + prefix + set of branches."""
    seed: int
    prefix_actions: np.ndarray   # (T_prefix, A)
    prefix_observations: np.ndarray  # (T_prefix, H, W, C)
    prefix_rewards: np.ndarray       # (T_prefix,)
    branches: Dict[str, BranchResult]


def run_prefix(
    env: Env,
    seed: int,
    prefix_actions: np.ndarray,
    record: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    obs, _ = env.reset(seed=seed)
    obs_list: List[np.ndarray] = [] if record else None  # type: ignore
    reward_list: List[float] = [] if record else None     # type: ignore

    for t in range(len(prefix_actions)):
        action = prefix_actions[t]
        next_obs, reward, terminated, truncated, _ = env.step(action)
        if record:
            obs_list.append(obs.astype(np.uint8))
            reward_list.append(float(reward))
        obs = next_obs
        if terminated or truncated:
            break

    if record:
        return np.stack(obs_list), np.array(reward_list, dtype=np.float32)
    return np.array([]), np.array([])


def run_branch(
    env: Env,
    branch_actions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
    """Execute branch_actions from current env state.

    Returns (observations, rewards, terminated, truncated).
    Observations are recorded *after* each step (matching run_prefix).
    """
    obs_list: List[np.ndarray] = []
    reward_list: List[float] = []
    terminated = False
    truncated = False

    for t in range(len(branch_actions)):
        action = branch_actions[t]
        next_obs, reward, term, trunc, _ = env.step(action)
        obs_list.append(np.array(next_obs, dtype=np.uint8))
        reward_list.append(float(reward))
        terminated = term
        truncated = trunc
        if terminated or truncated:
            break

    return np.stack(obs_list), np.array(reward_list, dtype=np.float32), terminated, truncated


def verify_deterministic_replay(
    env_name: str,
    seed: int,
    actions: np.ndarray,
    expected_obs: np.ndarray,
    expected_rewards: np.ndarray,
    atol: float = 1e-5,
) -> List[str]:
    env: Env = make_env(env_name, render_mode="rgb_array")
    replayed_obs, replayed_rewards = run_prefix(env, seed, actions, record=True)
    env.close()

    issues: List[str] = []
    T = min(len(actions), len(expected_obs), len(replayed_obs))
    if T == 0:
        issues.append("No steps to compare")
        return issues

    if len(replayed_obs) != len(expected_obs):
        issues.append(f"Length mismatch: expected {len(expected_obs)}, got {len(replayed_obs)}")
        T = min(len(replayed_obs), len(expected_obs))

    obs_diff = np.abs(replayed_obs[:T].astype(np.float32) - expected_obs[:T].astype(np.float32))
    if obs_diff.max() > atol:
        issues.append(f"Max observation diff {obs_diff.max():.4f} > {atol}")

    rew_diff = np.abs(replayed_rewards[:T] - expected_rewards[:T]).max()
    if rew_diff > atol:
        issues.append(f"Max reward diff {rew_diff:.4f} > {atol}")

    return issues


def run_branch_experiment(
    seed: int,
    prefix_actions: np.ndarray,
    branches: Dict[str, np.ndarray],
    env_name: str = "car_racing",
) -> BranchExperiment:
    env: Env = make_env(env_name, render_mode="rgb_array")
    prefix_obs, prefix_rewards = run_prefix(env, seed, prefix_actions, record=True)

    branch_results: Dict[str, BranchResult] = {}
    for name, branch_acts in branches.items():
        obs_b, rew_b, term, trunc = run_branch(env, branch_acts)
        branch_results[name] = BranchResult(
            branch_name=name,
            actions=branch_acts,
            observations=obs_b,
            rewards=rew_b,
            terminated=term,
            truncated=trunc,
        )
        # Reset env back to anchor state
        env.reset(seed=seed)
        run_prefix(env, seed, prefix_actions, record=False)

    env.close()
    return BranchExperiment(
        seed=seed,
        prefix_actions=prefix_actions,
        prefix_observations=prefix_obs,
        prefix_rewards=prefix_rewards,
        branches=branch_results,
    )


def save_branch_experiment(
    exp: BranchExperiment,
    out_dir: Path,
    source_manifest_hash: str = "",
    git_commit: str = "",
    config_ref: str = "",
    env_version: str = "",
) -> Path:
    """Save a branch experiment under ``out_dir/branches/``.

    Returns path to the saved ``.npz`` file.
    """
    branch_dir = out_dir / "branches"
    branch_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    fname = f"branch_seed{exp.seed}_{hash(exp.prefix_actions.tobytes()) % 10000:04d}"
    npz_path = branch_dir / f"{fname}.npz"

    data = {
        "seed": np.array([exp.seed]),
        "prefix_actions": exp.prefix_actions,
        "prefix_observations": exp.prefix_observations,
        "prefix_rewards": exp.prefix_rewards,
    }
    for name, br in exp.branches.items():
        data[f"{name}_actions"] = br.actions
        data[f"{name}_observations"] = br.observations
        data[f"{name}_rewards"] = br.rewards
        data[f"{name}_terminated"] = np.array([br.terminated])
        data[f"{name}_truncated"] = np.array([br.truncated])

    np.savez_compressed(npz_path, **data)

    # Sidecar metadata
    branch_defs: Dict[str, Tuple[int, ...]] = {}
    for name, br in exp.branches.items():
        branch_defs[name] = br.actions.shape

    meta = BranchMetadata(
        seed=exp.seed,
        source_manifest_hash=source_manifest_hash,
        branch_definitions=branch_defs,
        env_version=env_version,
        created_at=timestamp,
        git_commit=git_commit,
        config_ref=config_ref,
    )
    meta_path = npz_path.with_suffix(".branch.json")
    with open(meta_path, "w") as f:
        json.dump(dataclasses.asdict(meta), f, indent=2, sort_keys=True)

    return npz_path
