import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import TypedDict, List
from numpy.typing import NDArray


from torch import Tensor


@dataclass(frozen=True)
class Rollout:
    real_obs: 	"NDArray[np.uint8]"		# real observations
    real_acts:  "NDArray[np.float32]"	# real actions
    
    latents:    List[Tensor]   			# hidden states from imagine_rollout
    sim_acts:    List[Tensor] 	# the full segment of real actions used for warmup
    rewards:    List[float]    			# per-step predicted rewards
    cum_reward: float          			# sum(rewards)


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