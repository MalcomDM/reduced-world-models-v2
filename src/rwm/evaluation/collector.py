"""Evaluation-only rollout collector.

Saves observations, actions, rewards, terminated, truncated separately.
Requires an existing seed manifest and a seed registered in it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from gymnasium import Env

from rwm.envs.env import make_env
from rwm.evaluation.schema import (
    SeedManifest,
    make_episode_metadata,
    save_episode_metadata,
    load_seed_manifest,
)


def _compute_manifest_hash(manifest_path: Path) -> str:
    h = hashlib.sha256()
    h.update(manifest_path.read_bytes())
    return h.hexdigest()[:16]


def collect_evaluation_episode(
    manifest_path: Path,
    seed: int,
    out_dir: Path,
    policy_fn: Callable,
    policy_name: str = "manual",
    max_steps: int = 1000,
    early_push: int = 0,
    idle_threshold: int = 100,
    git_commit: str = "",
    config_ref: str = "",
    operator: str = "",
    render_mode: str = "rgb_array",
    env_version: str = "",
    clock=None,
    fps: int = 60,
    running_check: Optional[Callable[[], bool]] = None,
    on_env_ready: Optional[Callable[[], None]] = None,
) -> Path:
    """Collect one evaluation episode on a fixed seed.

    Parameters
    ----------
    manifest_path:
        Path to an existing seed manifest JSON file.
    seed:
        Track seed (must be registered in the manifest).
    out_dir:
        Output directory (episodes saved under ``out_dir/<split>/``).
    policy_fn:
        Callable ``(obs: np.ndarray) -> np.ndarray`` returning an action.
    clock:
        Optional ``pygame.time.Clock`` for human-mode frame rate limiting.
    running_check:
        Optional ``() -> bool``; when it returns False, collection stops
        cleanly and the partial episode is saved.
    on_env_ready:
        Optional callback invoked after the environment has reset and created
        its human display.  Interactive policies use this to initialize input.

    Returns
    -------
    Path to the saved ``.npz`` file.
    """
    manifest = load_seed_manifest(manifest_path)
    manifest.assert_valid()
    seed_str = str(seed)
    if seed_str not in manifest.entries:
        raise ValueError(
            f"Seed {seed} not found in manifest at {manifest_path}. "
            f"Registered seeds: {list(manifest.entries.keys())}"
        )
    split = manifest.entries[seed_str]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ep_id = f"{split}_{seed}_{timestamp}"
    ep_dir = out_dir / split
    ep_dir.mkdir(parents=True, exist_ok=True)
    npz_path = ep_dir / f"{ep_id}.npz"

    obs_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []
    reward_list: list[float] = []
    terminated_list: list[bool] = []
    truncated_list: list[bool] = []

    total_reward = 0.0
    idle_counter = 0

    env: Optional[Env] = None
    try:
        env = make_env("car_racing", render_mode=render_mode)
        obs, _ = env.reset(seed=seed)
        if on_env_ready is not None:
            on_env_ready()
        for t in range(max_steps):
            if running_check is not None and not running_check():
                break

            if t < early_push:
                action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                action = policy_fn(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)

            obs_list.append(obs.astype(np.uint8))
            action_list.append(action)
            reward_list.append(float(reward))
            terminated_list.append(bool(terminated))
            truncated_list.append(bool(truncated))

            obs = next_obs
            total_reward += float(reward)
            idle_counter = idle_counter + 1 if float(reward) < 0.1 else 0

            if terminated or truncated or idle_counter >= idle_threshold:
                break

            if clock is not None:
                clock.tick(fps)
    finally:
        if obs_list:
            np.savez_compressed(
                npz_path,
                obs=np.stack(obs_list), action=np.stack(action_list),
                reward=np.array(reward_list, dtype=np.float32),
                terminated=np.array(terminated_list, dtype=bool),
                truncated=np.array(truncated_list, dtype=bool),
            )
            meta = make_episode_metadata(
                track_seed=seed, split=split, policy=policy_name,
                git_commit=git_commit, config_ref=config_ref,
            )
            meta = dataclasses.replace(
                meta, steps=len(obs_list), terminated=any(terminated_list),
                truncated=any(truncated_list), operator=operator,
                max_steps=max_steps, early_push=early_push,
                idle_threshold=idle_threshold, render_mode=render_mode,
                env_version=env_version,
                manifest_hash=_compute_manifest_hash(manifest_path),
                manifest_path=str(manifest_path.resolve()),
            )
            save_episode_metadata(meta, npz_path.with_suffix(".episode.json"))
        if env is not None:
            env.close()

    if not obs_list:
        raise RuntimeError("No frames collected — episode too short.")

    return npz_path
