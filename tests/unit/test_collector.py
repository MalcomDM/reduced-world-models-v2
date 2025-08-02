import json, pytest
from pathlib import Path

from rwm.types import RolloutInfo, build_rollout_info
from rwm.data.collector import define_output_file_name, save_rollout_info


@pytest.mark.collector
def test_define_output_file_name(tmp_path: Path):

    scenario = "basic_turn"
    controller = "random"
    file_path = define_output_file_name(scenario, controller, tmp_path)

    assert file_path.parent.name == scenario
    assert controller in file_path.name
    assert file_path.suffix == ".npz"
    

@pytest.mark.collector
def test_build_rollout_info():

    info: RolloutInfo = build_rollout_info("u_curve", "controller", 45.0, 123, True)

    assert isinstance(info, dict)
    assert set(info.keys()) == {"scenario", "controller", "total_reward", "steps", "success"}
    assert info["scenario"] == "u_curve"
    assert info["success"] is True
    

@pytest.mark.collector
def test_save_and_load_info_json(tmp_path: Path):

    info = build_rollout_info("x", "random", 10.0, 50, True)
    file_path = tmp_path / "test.npz"

    save_rollout_info(file_path, info)

    with open(file_path.with_suffix(".info.json")) as f:
        loaded = json.load(f)

    assert loaded["controller"] == "random"
    assert loaded["success"] is True