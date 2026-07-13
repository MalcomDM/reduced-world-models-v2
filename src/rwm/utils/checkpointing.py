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
