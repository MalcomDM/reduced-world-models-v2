import time
from tqdm import trange
from pathlib import Path
from typing import Callable, Any

import numpy as np
from numpy.typing import NDArray
from gymnasium import Env

from rwm.envs.env import make_env
from rwm.types import RolloutInfo, build_rollout_info


def collect_rollout(
	env_name: str,
	policy_fn: Callable[[NDArray[np.float32]], NDArray[np.float32]],
	scenario_name: str,
	out_dir: Path,
	max_steps: int = 1000,
	idle_threshold: int = 100,
	render_mode: str = "rgb_array",
	early_push: int = 0,
) -> Path:
	env: Env[Any, Any] = make_env(env_name=env_name, render_mode=render_mode)
	obs, _ = env.reset()

	obs_list: list[NDArray[np.uint8]] = []
	action_list: list[NDArray[np.float32]] = []
	reward_list: list[float] = []
	done_list: list[bool] = []

	total_reward: float = 0.0
	idle_counter: int = 0

	for t in trange(max_steps, desc=f"Collecting {scenario_name}"):
		if t < early_push:
			action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
		else:
			action = policy_fn(obs)
		next_obs, reward, done, truncated, _ = env.step(action)

		obs_list.append(obs.astype(np.uint8))
		action_list.append(action)
		reward_list.append(float(reward))
		done_list.append(done or truncated)

		obs = next_obs
		total_reward += float(reward)
		idle_counter = idle_counter + 1 if float(reward) < 0.1 else 0

		if done or truncated or idle_counter >= idle_threshold:
			break

	env.close()

	if not obs_list:
		raise RuntimeError("No frames collected — rollout too short or invalid.")

	controller_name = getattr(policy_fn, "__name__", "controller")
	file_path = define_output_file_name(scenario_name, controller_name, out_dir)

	np.savez_compressed(
		file_path,
		obs=np.stack(obs_list),
		action=np.stack(action_list),
		reward=np.stack(reward_list),
		done=np.stack(done_list)
	)

	info_dict = build_rollout_info(
		scenario_name, controller_name, total_reward,
		steps=len(obs_list),
		success=idle_counter < idle_threshold
	)
	save_rollout_info( file_path, info_dict )

	print(f"\n✅ Saved rollout: {file_path.relative_to(out_dir)}")
	return file_path


def define_output_file_name( scenario: str, controller_name: str, out_dir: Path ) -> Path:
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    rollout_dir = out_dir / scenario
    rollout_dir.mkdir(parents=True, exist_ok=True)
    return rollout_dir / f"{timestamp}_{controller_name}.npz"


def save_rollout_info(file_path: Path, info_dict: RolloutInfo) -> None:
	info_path = file_path.with_suffix(".info.json")
	with open(info_path, "w") as f:
		import json
		json.dump(info_dict, f, indent=2)
