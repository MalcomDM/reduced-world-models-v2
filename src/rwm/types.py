from enum import Enum
from typing import TypedDict



class PolicyName( Enum):
    RANDOM 			= "random"
    RANDOM_SMOOTH 	= "random_smooth"
    HUMAN 			= "human"



class RolloutInfo(TypedDict):
    scenario: str
    controller: str
    total_reward: float
    steps: int
    success: bool


def build_rollout_info(
	scenario: str, controller: str, reward: float, steps: int, success: bool 
) -> RolloutInfo:
	return {
		"scenario": scenario,
		"controller": controller,
		"total_reward": reward,
		"steps": steps,
		"success": success
	}


from typing import TypedDict
from torch import Tensor


class RolloutSample(TypedDict):
	obs: Tensor         # shape (T, C, H, W)
	action: Tensor      # shape (T, 3)
	reward: Tensor      # shape (T,)
	done: Tensor        # shape (T,)