"""Structured checkpoint save/load with legacy compatibility.

Checkpoint format (version 2)::

    {
        "schema_version": 2,
        "model_state": { ... },
        "optimizer_state": { ... } | None,
        "scheduler_state": { ... } | None,
        "global_step": int,
        "epoch": int,
        "config": { ... },          # ExperimentConfig.to_dict()
        "metrics": { ... },         # last evaluation metrics
        "rng_state": {
            "python": ...,
            "numpy": ...,
            "torch": ...,
        },
        "dataset_manifest_ref": str | None,
    }

Legacy checkpoints (version 1, bare ``state_dict``) are detected and
loaded with a warning.
"""

from __future__ import annotations

import json
import pickle
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from rwm.config.experiment_config import ExperimentConfig

_CHECKPOINT_SCHEMA_VERSION = 2
_LEGACY_VERSIONS = (None, 1)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model_state: Dict[str, Any],
    config: ExperimentConfig,
    optimizer_state: Optional[Dict[str, Any]] = None,
    scheduler_state: Optional[Dict[str, Any]] = None,
    global_step: int = 0,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    dataset_manifest_ref: Optional[str] = None,
) -> Path:
    """Save a structured checkpoint.

    Parameters
    ----------
    path:
        Destination path (``.pt`` extension added if missing).
    model_state:
        ``model.state_dict()``.
    config:
        Resolved experiment configuration.
    optimizer_state:
        ``optimizer.state_dict()`` (optional).
    scheduler_state:
        Scheduler state dict (optional).
    global_step, epoch:
        Training progress counters.
    metrics:
        Last evaluation metrics dict.
    dataset_manifest_ref:
        Optional reference to a dataset manifest filename or ID.

    Returns
    -------
    ``path`` with ``.pt`` suffix.
    """
    path = path.with_suffix(".pt")

    # Capture RNG states
    rng_state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }

    checkpoint: Dict[str, Any] = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "global_step": global_step,
        "epoch": epoch,
        "config": config.to_dict(),
        "metrics": metrics or {},
        "rng_state": rng_state,
        "dataset_manifest_ref": dataset_manifest_ref,
    }

    torch.save(checkpoint, path)
    return path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: Path,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a checkpoint, supporting both structured (v2) and legacy
    (bare ``state_dict``) formats.

    Parameters
    ----------
    path:
        Path to the ``.pt`` file.
    map_location:
        Device map (e.g. ``"cpu"``, ``"cuda:0"``).  If ``None``, uses
        the checkpoint's original device.

    Returns
    -------
    Dict with at least:
        - ``"schema_version"``
        - ``"model_state"``
        - ``"config"`` (``ExperimentConfig`` or ``None`` for legacy)
        - ``"legacy"`` (``True`` if this is a bare state_dict)
    """
    path = path.with_suffix(".pt")
    raw = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(raw, dict) and "schema_version" in raw:
        # Structured checkpoint
        return _postprocess_structured(raw)
    else:
        # Legacy bare state_dict
        warnings.warn(
            f"Loading legacy checkpoint (bare state_dict) from {path}. "
            "Consider re-saving with save_checkpoint() for full metadata.",
            UserWarning,
            stacklevel=2,
        )
        return {
            "schema_version": 1,
            "model_state": raw if isinstance(raw, dict) else {},
            "config": None,
            "optimizer_state": None,
            "scheduler_state": None,
            "global_step": 0,
            "epoch": 0,
            "metrics": {},
            "rng_state": None,
            "dataset_manifest_ref": None,
            "legacy": True,
        }


# ---------------------------------------------------------------------------
# Model factory from checkpoint config
# ---------------------------------------------------------------------------

_VALID_REWARD_HEAD_KINDS = ("linear", "nonlinear")


_VALID_TEMPORAL_BACKENDS = ("causal_transformer", "minimal_sru")


def _resolve_temporal_config(cfg: Any) -> "TemporalConfig":
    """Extract ``TemporalConfig`` from a checkpoint config with defaults."""
    from rwm.config.experiment_config import TemporalConfig

    if cfg is None:
        return TemporalConfig()

    tc = None
    if hasattr(cfg, "temporal"):
        tc = cfg.temporal
    elif isinstance(cfg, dict) and "temporal" in cfg:
        tc = TemporalConfig.from_dict(cfg["temporal"])

    if tc is not None:
        return tc
    return TemporalConfig()


def _check_backend_compatibility(
    temporal_cfg: "TemporalConfig",
    state_dict: Dict[str, Any],
) -> None:
    """Raise ``ValueError`` if the backend and state_dict keys are incompatible.

    Detection rules:
    - If state_dict contains ``world_hd.projection.*`` → SRU checkpoint.
    - If state_dict contains ``world_hd.input_proj.*`` → causal checkpoint.
    - If config backend disagrees with detected keys → error.
    - If no config (legacy) or no temporal keys → silently accept (auto-detect
      will use the config default or the detected keys in a future enhancement).
    """
    has_causal_keys = any(k.startswith("world_hd.input_proj") or
                          k.startswith("world_hd.pos_emb") or
                          k.startswith("world_hd.encoder") for k in state_dict)
    has_sru_keys = any(k.startswith("world_hd.projection") for k in state_dict)

    if has_causal_keys and temporal_cfg.backend == "minimal_sru":
        raise ValueError(
            "Architecture mismatch: checkpoint contains causal Transformer "
            "world_hd keys (e.g., 'world_hd.input_proj.weight') but config "
            f"specifies backend='{temporal_cfg.backend}'. "
            "Use backend='causal_transformer' to load this checkpoint."
        )
    if has_sru_keys and temporal_cfg.backend == "causal_transformer":
        raise ValueError(
            "Architecture mismatch: checkpoint contains SRU world_hd keys "
            "(e.g., 'world_hd.projection.weight') but config specifies "
            f"backend='{temporal_cfg.backend}'. "
            "Use backend='minimal_sru' to load this checkpoint."
        )

    # Legacy checkpoints with no temporal backend info: auto-detect from keys.
    if temporal_cfg.backend == "causal_transformer" and has_sru_keys:
        raise ValueError(
            "Architecture mismatch: checkpoint contains SRU world_hd keys "
            "(e.g., 'world_hd.projection.weight') but no temporal backend "
            "was specified.  Use backend='minimal_sru' to load this checkpoint."
        )
    if temporal_cfg.backend == "minimal_sru" and has_causal_keys:
        raise ValueError(
            "Architecture mismatch: checkpoint contains causal Transformer "
            "world_hd keys (e.g., 'world_hd.input_proj.weight') but config "
            f"specifies backend='{temporal_cfg.backend}'. "
            "Use backend='causal_transformer' to load this checkpoint."
        )


def model_from_checkpoint(
    checkpoint: Dict[str, Any],
    action_dim: int = 3,
    tokenizer_eval_mode_override: Optional[str] = None,
) -> "torch.nn.Module":
    """Build a ``ReducedWorldModel`` from checkpoint metadata.

    Reads reward-head, selection, and temporal-backend configuration from
    the checkpoint's ``config``.  Legacy/missing fields default to
    ``"linear"`` head, ``"learned"`` selection with ``k=8``, and
    ``"causal_transformer"`` temporal backend.

    Cross-backend loading raises a clear ``ValueError`` — a causal checkpoint
    must never be silently guessed as SRU, and vice versa.

    Parameters
    ----------
    tokenizer_eval_mode_override:
        If provided (``"sample"`` or ``"mean"``), overrides the checkpoint's
        saved tokenizer evaluation policy at runtime.  If ``None``, the
        checkpoint-saved policy is used.
    """
    from rwm.models.rwm.model import ReducedWorldModel

    if tokenizer_eval_mode_override is not None:
        assert tokenizer_eval_mode_override in (
            "sample", "mean",
        ), f"tokenizer_eval_mode_override must be 'sample' or 'mean', got {tokenizer_eval_mode_override!r}"

    kind = "linear"
    hidden = 32
    sel_mode = "learned"
    sel_k = 8
    sel_seed = 0
    tok_eval_mode = "sample"

    cfg = checkpoint.get("config")
    temporal_cfg = _resolve_temporal_config(cfg)

    if cfg is not None:
        # Controller config
        ctrl_cfg = None
        if hasattr(cfg, "controller"):
            ctrl_cfg = cfg.controller
        elif isinstance(cfg, dict) and "controller" in cfg:
            from rwm.config.experiment_config import ControllerConfig
            ctrl_cfg = ControllerConfig.from_dict(cfg["controller"])
        if ctrl_cfg is not None:
            kind = getattr(ctrl_cfg, "reward_head_kind", "linear")
            hidden = getattr(ctrl_cfg, "reward_head_hidden_dim", 32)

        # Perception / selection config
        pc = None
        if hasattr(cfg, "perception"):
            pc = cfg.perception
        elif isinstance(cfg, dict) and "perception" in cfg:
            from rwm.config.experiment_config import PerceptionConfig
            pc = PerceptionConfig.from_dict(cfg["perception"])
        if pc is not None:
            sel_mode = getattr(pc, "selection_mode", "learned")
            sel_k = getattr(pc, "k", 8)
            sel_seed = getattr(pc, "selection_seed", 0)
            tok_eval_mode = getattr(pc, "tokenizer_eval_mode", "sample")

    # Apply override if provided
    if tokenizer_eval_mode_override is not None:
        tok_eval_mode = tokenizer_eval_mode_override

    if kind not in _VALID_REWARD_HEAD_KINDS:
        raise ValueError(
            f"Invalid reward_head_kind in checkpoint metadata: {kind!r}. "
            f"Must be one of {_VALID_REWARD_HEAD_KINDS}."
        )
    if not isinstance(hidden, int) or hidden < 1:
        raise ValueError(
            f"Invalid reward_head_hidden_dim in checkpoint metadata: {hidden}. "
            "Must be a positive integer."
        )

    # Validate temporal backend compatibility before building the model.
    state_dict = checkpoint.get("model_state", {})
    _check_backend_compatibility(temporal_cfg, state_dict)

    model = ReducedWorldModel(
        action_dim=action_dim,
        reward_head_kind=kind,
        reward_head_hidden_dim=hidden,
        selection_mode=sel_mode,
        selection_k=sel_k,
        selection_seed=sel_seed,
        tokenizer_eval_mode=tok_eval_mode,
        temporal_config=temporal_cfg,
    )
    model.load_state_dict(checkpoint["model_state"])
    return model


def _postprocess_structured(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert structured checkpoint raw dict to standardised output."""
    config_dict = raw.get("config")
    config: Optional[ExperimentConfig] = None
    if config_dict is not None:
        try:
            config = ExperimentConfig.from_dict(config_dict)
        except Exception as exc:
            warnings.warn(
                f"Failed to deserialize config: {exc}", stacklevel=2,
            )

    return {
        "schema_version": raw.get("schema_version", _CHECKPOINT_SCHEMA_VERSION),
        "model_state": raw.get("model_state", {}),
        "config": config,
        "optimizer_state": raw.get("optimizer_state"),
        "scheduler_state": raw.get("scheduler_state"),
        "global_step": raw.get("global_step", 0),
        "epoch": raw.get("epoch", 0),
        "metrics": raw.get("metrics", {}),
        "rng_state": raw.get("rng_state"),
        "dataset_manifest_ref": raw.get("dataset_manifest_ref"),
        "legacy": False,
    }
