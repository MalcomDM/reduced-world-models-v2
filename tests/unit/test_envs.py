import pytest
from gymnasium import Env
import numpy as np

from rwm.envs.env import make_env


@pytest.mark.envs
def test_make_env_success():
    env = make_env("car_racing")
    assert isinstance(env, Env)
    obs, _ = env.reset()
    assert isinstance(obs, (np.ndarray, list))
    env.close()


@pytest.mark.envs
def test_make_env_invalid_raises():
    with pytest.raises(ValueError):
        make_env("invalid_env_name")