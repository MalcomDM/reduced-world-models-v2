import pytest
import torch

from rwm.policies.controller_policy import ControllerPolicy
from rwm.models.rwm.model import ReducedWorldModel
from rwm.models.controller.model import Controller

@pytest.fixture
def dummy_world_model() -> ReducedWorldModel:
    model = ReducedWorldModel()
    # No observational dropout during tests
    if hasattr(model, "obs_drop"):
        model.obs_drop.p = 0.0
    model.eval()
    return model

@pytest.fixture
def dummy_controller():
	return Controller()


@pytest.mark.actions
def test_act_from_rwm_state_bounds(dummy_world_model: ReducedWorldModel,
									dummy_controller: Controller):
	policy = ControllerPolicy(dummy_controller, noise_std=0.0)

	# New latent/state size comes from the causal block
	d_model = dummy_world_model.world_hd.d_model

	# Build a zero world state of the correct size
	h = torch.zeros(1, d_model, dtype=torch.float32)
	a = policy.act_from_rwm_state(h)

	# Output must be a tensor of shape (1,3) within valid action bounds
	assert isinstance(a, torch.Tensor)
	assert a.shape == (1, 3)

	steer, gas, brk = a[0]
	assert -1.0 <= float(steer) <= 1.0
	assert  0.0 <= float(gas)   <= 1.0
	assert  0.0 <= float(brk)   <= 1.0