import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import TypedDict, List, Optional, NamedTuple
from numpy.typing import NDArray


from torch import Tensor


# ---------------------------------------------------------------------------
# WorldModelOutput — structured forward output
# ---------------------------------------------------------------------------

class WorldModelOutput(NamedTuple):
    """Structured output of ``ReducedWorldModel.forward()``.

    This replaces the anonymous 6-element tuple returned previously.
    Every field has a defined shape (batch-first) and semantic meaning.

    Fields
    ------
    world_state:   ``(B, D)`` — latent temporal state s_{{t+1}}.
    reward_pred:   ``(B, 1)`` — predicted immediate reward r_{{t+1}}.
    mask_soft:     ``(B, N)`` — soft Top-K mask over patch tokens.
    indices:       ``(B, K)`` — selected patch token indices.
    history:       ``(B, T', input_dim)`` — updated temporal history buffer.
    lengths:       ``(B,)`` — valid lengths in ``history``.
    tok_mu:        ``(B, N_patches, D_token)`` or ``None`` — tokenizer mean.
    tok_logvar:    ``(B, N_patches, D_token)`` or ``None`` — tokenizer log-variance.
    """
    world_state: Tensor
    reward_pred: Tensor
    mask_soft: Tensor
    indices: Tensor
    history: Tensor
    lengths: Tensor
    tok_mu: Optional[Tensor] = None
    tok_logvar: Optional[Tensor] = None
    reward_pred_seq: Optional[Tensor] = None  # (B, T) per-step predictions from forward_sequence


# ---------------------------------------------------------------------------
# Rollout (legacy imagination result)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rollout:
    """Imagined rollout from RolloutSimulator (controller training path).

    Index `i` in `latents[i]` / `sim_acts[i]` / `rewards[i]` corresponds
    to the i-th imagined step **after** warmup.

    Fields
    ------
    real_obs:  Real observation segment used for warmup (T_warm, H, W, C).
    real_acts: Real action segment used for warmup (T_warm, A).
    latents:   World states after each imagined step.
    sim_acts:  Controller actions sampled during imagination.
    rewards:   Predicted reward per imagined step (Python float).
    cum_reward: Sum of `rewards`.
    """
    real_obs: 	"NDArray[np.uint8]"
    real_acts:  "NDArray[np.float32]"
    latents:    List[Tensor]
    sim_acts:    List[Tensor]
    rewards:    List[float]
    cum_reward: float


# ---------------------------------------------------------------------------
# Policy names
# ---------------------------------------------------------------------------

class PolicyName(Enum):
    RANDOM 			= "random"
    RANDOM_SMOOTH 	= "random_smooth"
    HUMAN 			= "human"


# ---------------------------------------------------------------------------
# Rollout metadata (saved sidecar)
# ---------------------------------------------------------------------------

class RolloutInfo(TypedDict):
    """Metadata sidecar for a single `.npz` rollout file.

    Saved as `{filename}.info.json` alongside the `.npz` rollout data.
    """
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


# ---------------------------------------------------------------------------
# Training-sample types
# ---------------------------------------------------------------------------

class RolloutSample(TypedDict):
    """A single sequence window sampled from a rollout file.

    Shapes assume batch dimension B is absent (single sample).

    Transition semantics (per index ``t``):
        obs[t]    = s_t           observation before action
        action[t] = a_t           action selected from s_t
        reward[t] = r_{t+1}       reward from env.step(a_t)
        done[t]   = terminal_{t+1} whether s_{t+1} is terminal

    Reference: `docs/transition_contract.md`
    """
    obs: Tensor         # shape (T, C, H, W)
    action: Tensor      # shape (T, A)   where A = 3 (steer, gas, brake)
    reward: Tensor      # shape (T,)
    done: Tensor        # shape (T,)  bool


class ProbeSet(TypedDict):
    """A fixed set of observations and actions for deterministic evaluation.

    Used to measure latent drift across training runs or architectural changes.
    Fields are kept in CPU memory and can be moved to device at test time.
    """
    obs: Tensor         # (N, C, H, W)   N probe frames
    action: Tensor      # (N, A)         N probe actions


# ---------------------------------------------------------------------------
# Episode-safe split types
# ---------------------------------------------------------------------------

class EpisodeSamples(TypedDict):
    """All samples belonging to one episode (rollout .npz file)."""
    file_path: str
    obs: Tensor             # (T, C, H, W)
    action: Tensor          # (T, A)
    reward: Tensor          # (T,)
    done: Tensor            # (T,)  bool


class TrainValSplit(TypedDict):
    """Train/validation split by episode (no overlapping windows)."""
    train: List[EpisodeSamples]
    val: List[EpisodeSamples]