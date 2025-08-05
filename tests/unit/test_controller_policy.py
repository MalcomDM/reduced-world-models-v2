import pytest
import torch

from rwm.policies.controller_policy import ControllerPolicy
from rwm.models.rwm_deterministic.model import ReducedWorldModel
from rwm.models.controller.model import Controller

@pytest.fixture
def dummy_world_model():
	# world_rnn.dropout_prob=0 so no stochastic masking
	model = ReducedWorldModel()
	model.world_rnn.dropout_prob = 0.0
	return model

@pytest.fixture
def dummy_controller():
	return Controller()


@pytest.mark.actions
def test_act_from_rwm_state_bounds(dummy_world_model: ReducedWorldModel, dummy_controller: Controller):
	policy = ControllerPolicy(dummy_controller, noise_std=0.0)
	hidden_size = dummy_world_model.world_rnn.rnn_cell.hidden_size
	h = torch.zeros(1, hidden_size)
	a = policy.act_from_rwm_state(h)

	# Output must be a tensor of shape (1,3) within valid action bounds
	assert isinstance(a, torch.Tensor)
	assert a.shape == (1, 3)

	steer, gas, brk = a[0]
	assert -1.0 <= steer <= 1.0
	assert  0.0 <= gas   <= 1.0
	assert  0.0 <= brk   <= 1.0