import os, pytest
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor

from rwm.types import Rollout
from rwm.utils.behavior_memory import BehaviorMemory



@pytest.fixture
def scratch_dir(tmp_path: Path) -> str:
    return str(tmp_path / "situations")


@pytest.fixture
def dummy_rollout() -> Rollout:
    # Create a trivial rollout: empty latents/actions, reward=1.0
    r_o = np.zeros((3, 64, 64, 3), dtype=np.uint8)
    r_a = np.zeros((3, 3), dtype=np.float32)
    lat = [torch.zeros(1, 16) for _ in range(3)]
    act = [torch.zeros(1, 3) for _ in range(3)]
    rew = [0.5, 0.5, 0.5]
    return Rollout(r_o, r_a, lat, act, rew, sum(rew))


def make_obs_act() -> Tuple[NDArray[np.uint8], NDArray[np.float32]]:
    # 5 frames of 64×64×3 zeros and zero actions
    obs = np.zeros((5,64,64,3), dtype=np.uint8)
    acts = np.zeros((5,3), dtype=np.float32)
    return obs, acts


@pytest.mark.memory
def test_add_and_len_and_file_creation(scratch_dir: str, dummy_rollout: Rollout):
    bm = BehaviorMemory(scratch_dir, max_size=10)
    obs, acts = make_obs_act()
    
    h0 = torch.zeros(1, 16)					# initial H state
    bm.add(h0, dummy_rollout.cum_reward, obs, acts)
    
    assert len(bm) == 1						# memory has one entry
    
    key = next(iter(bm.states))				# corresponding .npz file exists
    path = bm.get_situation_path(key)
    assert os.path.isfile(path)
    data = np.load(path)
    assert "obs" in data and "actions" in data
    data.close()


@pytest.mark.memory
def test_replacement_of_worse_rollout(scratch_dir: str):
    bm = BehaviorMemory(scratch_dir, max_size=1)
    obs, acts = make_obs_act()
    h0 = torch.zeros(1, 16)
    
    bm.add(h0, 1.5, obs, acts)									# Add first rollout with reward=1.5
    bm.add(h0, 1.0, obs, acts)									# Add a worse rollout for same key
    
    assert bm.states[bm.hash_state(h0)]["cum_reward"] == 1.5	# Should still only have the better one


@pytest.mark.memory
def test_sample_and_max_size(scratch_dir: str, dummy_rollout: Rollout):
    bm = BehaviorMemory(scratch_dir, max_size=2)
    obs, acts = make_obs_act()
    
    for i in range(2):								# Add two distinct keys
        h = torch.ones(1, 16) * i
        bm.add(h, dummy_rollout.cum_reward, obs, acts)
    
    samples = bm.sample(5)							# sample more than available
    assert len(samples) == 2


@pytest.mark.memory
def test_save_and_load(tmp_path: Path, scratch_dir: str, dummy_rollout: Rollout):
    bm = BehaviorMemory(scratch_dir, max_size=10)
    obs, acts = make_obs_act()
    h0 = torch.zeros(1, 16)
    bm.add(h0, dummy_rollout.cum_reward, obs, acts)

    file = str(tmp_path / "bm.pkl")
    bm.save(file)
    
    bm2 = BehaviorMemory(scratch_dir, max_size=10)	# clear and reload
    bm2.load(file)
    assert len(bm2) == 1
    assert bm2.states == bm.states


@pytest.mark.memory
def test_recompute_keys(scratch_dir: str, dummy_rollout: Rollout):
	bm = BehaviorMemory(scratch_dir, max_size=10)
	obs, acts = make_obs_act()

	# Two distinct starting latents
	h0 = torch.zeros(1, 16)
	h1 = torch.ones(1, 16)
	bm.add(h0, dummy_rollout.cum_reward, obs, acts)
	bm.add(h1, dummy_rollout.cum_reward, obs, acts)
	old_keys = set(bm.states.keys())
	assert len(old_keys) == 2

	# Dummy model that maps any obs-> same h=ones
	class DummyModel(torch.nn.Module):
		def __init__(self):
			super().__init__()		# type: ignore
			self.dummy = torch.nn.Parameter(torch.zeros(1))
			self.world_rnn = type("", (), {"rnn_cell": type("", (), {"hidden_size":16})()})()
               
		def forward(
				self, img:Tensor, a:Tensor, h:Tensor, c:Tensor, force_keep_input:bool
		) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
			# forces h to uniform ones
			return torch.ones_like(h), c, c, c, c
	model = DummyModel()

	# recompute: all old keys map to same new key
	bm.recompute_keys(model)  # dummy preprocess
	new_keys = set(bm.states.keys())
	assert len(new_keys) == 1

	# reencode_count should have incremented
	info = next(iter(bm.states.values()))
	assert info["reencode_count"] == 2

