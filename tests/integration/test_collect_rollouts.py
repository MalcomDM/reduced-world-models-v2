import json, pytest
import numpy as np
from pathlib import Path
from numpy.typing import NDArray

from rwm.policies.random_policy import RandomPolicy
from rwm.data.collector import collect_rollout

def always_idle_policy(_: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)


def minimal_reward_policy(_: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)


def assertive_policy(_: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.array([0.9, 0.0, 0.0], dtype=np.float32) 



@pytest.mark.integration
def test_collect_rollout_creates_file(tmp_path: Path):

    policy = RandomPolicy()
    file_path = collect_rollout(
        env_name="car_racing",
        policy_fn=policy.act,
        scenario_name="test_scenario",
        out_dir=tmp_path,
        max_steps=5,
        idle_threshold=5,
    )

    assert file_path.exists()
    assert file_path.with_suffix(".info.json").exists()


@pytest.mark.integration
def test_collect_rollout_stops_at_idle_threshold(tmp_path: Path):
    idle_threshold = 10
    max_steps = 100

    file_path = collect_rollout(
        env_name="car_racing",
        policy_fn=always_idle_policy,
        scenario_name="idle_test",
        out_dir=tmp_path,
        max_steps=max_steps,
        idle_threshold=idle_threshold,
    )

    with np.load(file_path) as data:
        rewards = data["reward"]
        steps = len(rewards)

    assert steps < max_steps, "Idle logic failed — rollout ran full length"


@pytest.mark.integration
def test_rollout_info_success_flag(tmp_path: Path):
    file_path = collect_rollout(
        env_name="car_racing",
        policy_fn=minimal_reward_policy,
        scenario_name="info_test",
        out_dir=tmp_path,
        max_steps=10,
        idle_threshold=5,
    )

    info_path = file_path.with_suffix(".info.json")
    with open(info_path) as f:
        info = json.load(f)

    assert isinstance(info, dict)
    assert info["scenario"] == "info_test"
    assert info["controller"] == "minimal_reward_policy"
    assert "success" in info
    assert info["success"] is False
    

@pytest.mark.integration
def test_collect_rollout_early_push_applied(tmp_path: Path):
    early_push = 3
    file_path = collect_rollout(
        env_name="car_racing",
        policy_fn=assertive_policy,
        scenario_name="early_push_test",
        out_dir=tmp_path,
        max_steps=6,
        idle_threshold=100,
        early_push=early_push
    )

    with np.load(file_path) as data:
        actions = data["action"]
        early_actions = actions[:early_push]
        for a in early_actions:
            np.testing.assert_array_almost_equal(a, [0.0, 1.0, 0.0], decimal=3)