import logging
import gymnasium as gym
from gymnasium import Env
from typing import Any


_LOGGER = logging.getLogger(__name__)
_ENV_REGISTRY: dict[str,str] = {
	"car_racing": "CarRacing-v3",
	"car_racing_dream": "CarRacingDream-v0"
}


def make_env(
    env_name: str = "car_racing",
    render_mode: str = "rgb_array"
) -> Env[Any, Any]:
	if env_name not in _ENV_REGISTRY:
		raise ValueError( f"Unknown env_name '{env_name}'. Avaiable: {list(_ENV_REGISTRY.keys())}" )
	
	env_id: str = _ENV_REGISTRY[env_name]
	try:
		env: Env[Any, Any] = gym.make(env_id, render_mode=render_mode)  # type: ignore
		return env
	except Exception as e:
		_LOGGER.error(f"Failed to create env '{env_name}: {e}")
		raise