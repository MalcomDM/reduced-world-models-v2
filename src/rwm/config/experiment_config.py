"""Typed, serializable experiment configuration.

All defaults match the current project-wide constants defined in
``config.py``.  Config groups are plain dataclasses with a common
``to_dict()`` / ``from_dict()`` pattern so they round-trip through JSON
without external dependencies.
"""

import dataclasses
import json
import typing
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_dict(obj: Any) -> Dict[str, Any]:
    """Recursive dataclass → dict (handles nested dataclasses)."""
    return dataclasses.asdict(obj)


def _from_dict(cls: type, data: Dict[str, Any]) -> Any:
    """Recursive dict → dataclass (handles nested dataclasses via type hints)."""
    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs: Dict[str, Any] = {}
    for name, value in data.items():
        if name in hints:
            ft = hints[name]
            # If the field type is itself a dataclass
            origin = getattr(ft, "__origin__", None)
            if hasattr(ft, "__dataclass_fields__") and isinstance(value, dict):
                kwargs[name] = _from_dict(ft, value)
            elif (
                hasattr(ft, "__origin__")
                and ft.__origin__ is list
                and ft.__args__
                and hasattr(ft.__args__[0], "__dataclass_fields__")
                and isinstance(value, list)
            ):
                kwargs[name] = [_from_dict(ft.__args__[0], v) for v in value]
            else:
                kwargs[name] = value
    return cls(**kwargs)


def _serialize(obj: Any) -> str:
    """JSON string with sorted keys for deterministic output."""
    return json.dumps(_as_dict(obj), indent=2, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Config groups
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class DataConfig:
    """Rollout data and dataset parameters."""
    dataset_dir: str = "data/rollouts/rwm_deterministic"
    sequence_len: int = 16
    image_size: int = 64
    val_ratio: float = 0.2
    include_done_in_train: bool = False
    num_workers: int = 6
    pin_memory: bool = True
    cache_dir: str = ""  # path to pre-built frame cache (empty = no cache)

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        return _from_dict(cls, data)

    def to_json(self) -> str:
        return _serialize(self)


@dataclasses.dataclass(frozen=True)
class PerceptionConfig:
    """CNN encoder, tokenizer, attention scorer, and Top-K selector."""
    conv_filters: List[int] = dataclasses.field(default_factory=lambda: [32, 64, 16])
    conv_kernel_sizes: List[int] = dataclasses.field(default_factory=lambda: [4, 3, 1])
    conv_strides: List[int] = dataclasses.field(default_factory=lambda: [2, 1, 1])
    conv_paddings: List[int] = dataclasses.field(default_factory=lambda: [1, 1, 0])
    conv_activations: List[str] = dataclasses.field(
        default_factory=lambda: ["relu", "relu", "relu"]
    )
    patch_size: int = 4
    patch_stride: int = 2
    patch_padding: int = 0
    token_dim: int = 16
    k: int = 8
    query_dim: int = 16
    values_dim: int = 32
    selection_mode: str = "learned"
    selection_seed: int = 0
    tokenizer_eval_mode: str = "sample"

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionConfig":
        return _from_dict(cls, data)

    def to_json(self) -> str:
        return _serialize(self)


@dataclasses.dataclass(frozen=True)
class TemporalConfig:
    """Causal Transformer temporal world model."""
    seq_len: int = 20
    world_state_dim: int = 80
    observational_dropout: float = 0.6
    warmup_steps: int = 5
    ffn_mult: int = 2
    transformer_dropout: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalConfig":
        return _from_dict(cls, data)

    def to_json(self) -> str:
        return _serialize(self)


@dataclasses.dataclass(frozen=True)
class ControllerConfig:
    """Shared controller-trunk and head architecture configuration."""
    action_dim: int = 3
    hidden_dim: int = 80
    noise_std: float = 0.3
    rollout_len: int = 20
    n_rollouts: int = 100
    top_k: int = 10
    positive_threshold: float = 0.0
    memory_batch: int = 20
    memory_size: int = 1000
    reward_head_kind: str = "linear"
    reward_head_hidden_dim: int = 32

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ControllerConfig":
        return _from_dict(cls, data)

    def to_json(self) -> str:
        return _serialize(self)


@dataclasses.dataclass(frozen=True)
class TemporalMaskConfig:
    """Temporal observation-masking curriculum for D.1 training.

    When enabled, a contiguous block of steps after ``warmup_steps`` is
    masked (spatial rep → zero) with per-sample probability that ramps
    linearly from 0 to ``target_mask_probability`` over ``ramp_epochs``.
    """
    enabled: bool = False
    warmup_steps: int = 4
    horizons: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 4, 8, 12])
    target_mask_probability: float = 0.5
    ramp_epochs: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalMaskConfig":
        return _from_dict(cls, data)

    def to_json(self) -> str:
        return _serialize(self)


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Optimization and training loop."""
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_epochs: int = 50
    alpha: float = 1.0
    beta: float = 1.0
    warmup_steps: int = 20
    error_threshold: float = 0.35
    kl_beta: float = 1.0
    temporal_mask: TemporalMaskConfig = dataclasses.field(
        default_factory=TemporalMaskConfig,
    )

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        return _from_dict(cls, data)

    def to_json(self) -> str:
        return _serialize(self)


@dataclasses.dataclass(frozen=True)
class ActorCriticConfig:
    """Actor-Critic head calibration configuration (Stage 4.0)."""
    hidden_dim: int = 64
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.997
    lambda_: float = 0.95
    entropy_coef: float = 1e-3
    target_update_rate: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActorCriticConfig":
        return _from_dict(cls, data)

    def to_json(self) -> str:
        return _serialize(self)


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration.

    This is the single resolved config for a run.  All sub-configs are
    frozen and serializable.
    """
    experiment_name: str = "default"
    run_id: str = ""
    seed: int = 42
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    perception: PerceptionConfig = dataclasses.field(
        default_factory=PerceptionConfig
    )
    temporal: TemporalConfig = dataclasses.field(
        default_factory=TemporalConfig
    )
    controller: ControllerConfig = dataclasses.field(
        default_factory=ControllerConfig
    )
    actor_critic: ActorCriticConfig = dataclasses.field(
        default_factory=ActorCriticConfig,
    )
    training: TrainingConfig = dataclasses.field(
        default_factory=TrainingConfig
    )

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        return _from_dict(cls, data)

    def to_json(self) -> str:
        return _serialize(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
