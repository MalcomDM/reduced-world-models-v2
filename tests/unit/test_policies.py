import pytest
import numpy as np
from rwm.policies.random_policy import RandomPolicy


@pytest.mark.actions
def test_random_policy_action_shape_and_range():
    policy = RandomPolicy()
    obs = np.zeros((96, 96, 3), dtype=np.float32)

    action = policy.act(obs)
    assert action.shape == (3,)
    assert -1.0 <= action[0] <= 1.0
    assert 0.0 <= action[1] <= 1.0
    assert 0.0 <= action[2] <= 1.0


@pytest.mark.actions
def test_random_policy_smooth_transition():
    policy = RandomPolicy(smooth=True)
    obs = np.zeros((96, 96, 3), dtype=np.float32)

    a1 = policy.act(obs)
    a2 = policy.act(obs)

    assert a1.shape == (3,)
    assert a2.shape == (3,)
    assert not np.allclose(a1, a2)