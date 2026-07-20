"""Immutable factual-pointer record types and stable identity.

Record identity derives from the source-file SHA-256 and in-file
timestep, so it survives project-directory relocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 1
TIMING_CONTRACT_VERSION: str = "1.0"  # docs/contracts/transition_contract.md §11.1


# ---------------------------------------------------------------------------
# Stable record identity
# ---------------------------------------------------------------------------


def make_record_id(source_hash: str, timestep: int) -> str:
    """Deterministic, path-independent record identifier.

    Uses only the content-addressed file hash and the in-episode timestep.
    Moving or renaming the project directory does not change record IDs.
    Identical episodes (same bytewise NPZ content) produce identical IDs.
    """
    return f"{source_hash}:{timestep}"


# ---------------------------------------------------------------------------
# Immutable pointer record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactualPointer:
    """Immutable factual pointer — the permanent source of truth.

    All fields are frozen after construction.  Derived priority metadata
    is stored separately (see ``ArchiveEntry``).
    """

    schema_version: int = SCHEMA_VERSION

    # --- stable identity ---
    record_id: str = ""
    dataset_manifest: str = ""
    data_split_seed: int = 0

    # --- source provenance ---
    source_path: str = ""       # data-root-relative path
    source_hash: str = ""       # SHA-256 of the .npz
    episode_id: str = ""        # source file stem
    timestep: int = 0
    source_episode_length: int = 0

    # --- contract ---
    timing_contract_version: str = TIMING_CONTRACT_VERSION
    behavior_policy: Optional[str] = None

    # --- transition data (no observations) ---
    immediate_reward: float = 0.0
    legacy_done: bool = False
    terminated: Optional[bool] = None
    truncated: Optional[bool] = None

    # --- derived metrics (set after episode finalization) ---
    factual_return_H12: Optional[float] = None
    directional_change_h4: Optional[float] = None


# ---------------------------------------------------------------------------
# Archive entry — pointer + derived priority metadata
# ---------------------------------------------------------------------------


@dataclass
class ArchiveEntry:
    """A single archive row: immutable pointer plus versioned priority state.

    ``pointer`` is a ``FactualPointer`` (frozen, immutable).
    ``priority`` is a mutable dict that is replaced on every ``finalize()``.
    """

    pointer: FactualPointer
    priority: Dict[str, Any] = field(default_factory=dict)

    @property
    def record_id(self) -> str:
        return self.pointer.record_id


# ---------------------------------------------------------------------------
# Priority configuration snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorityConfig:
    """Canonical priority configuration for the Stage-7.0B index."""

    selected_H: int = 12
    selected_h: int = 4
    eta: float = 0.1
    lambda_pos: float = 1.0
    lambda_neg: float = 1.0
    lambda_up: float = 1.0
    lambda_down: float = 1.0
    lambda_legacy_done: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    rho: float = 0.25
    active_set_M: int = 1024
    quantize_decimals: int = 8

    def __post_init__(self) -> None:
        if self.selected_H < 1:
            raise ValueError("selected_H must be positive")
        if self.selected_h < 1:
            raise ValueError("selected_h must be positive")
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must be in [0, 1]")
        for name in (
            "lambda_pos",
            "lambda_neg",
            "lambda_up",
            "lambda_down",
            "lambda_legacy_done",
            "alpha",
            "beta",
            "rho",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.active_set_M < 1:
            raise ValueError("active_set_M must be positive")
        if self.quantize_decimals < 0:
            raise ValueError("quantize_decimals must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_H": self.selected_H,
            "selected_h": self.selected_h,
            "eta": self.eta,
            "lambda_pos": self.lambda_pos,
            "lambda_neg": self.lambda_neg,
            "lambda_up": self.lambda_up,
            "lambda_down": self.lambda_down,
            "lambda_legacy_done": self.lambda_legacy_done,
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "active_set_M": self.active_set_M,
            "quantize_decimals": self.quantize_decimals,
        }


CANONICAL_CONFIG = PriorityConfig()
