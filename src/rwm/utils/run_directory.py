"""Structured run-directory creation and management.

Directory layout::

    runs/<experiment_name>/<run_id>/
        config.json
        environment.json
        git_metadata.json
        metrics/
        checkpoints/
        probes/

``run_id`` is deterministic when explicitly provided, otherwise a
timestamp-based unique string.
"""

from __future__ import annotations

import datetime
import dataclasses
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from rwm.config.experiment_config import ExperimentConfig

_RUNS_ROOT = Path("runs")


def _timestamp_run_id() -> str:
    """Collision-safe timestamp run ID (microsecond precision)."""
    return datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")


def _dictify_config(cfg: ExperimentConfig) -> Dict[str, Any]:
    return cfg.to_dict()


def _gather_environment() -> Dict[str, Any]:
    """Collect Python, PyTorch, CUDA, and platform details."""
    env: Dict[str, Any] = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["cuda_device_count"] = torch.cuda.device_count()
        env["cuda_device_name"] = torch.cuda.get_device_name(0)
        try:
            env["cuda_version"] = torch.version.cuda
        except Exception:
            pass
    return env


def _gather_git_metadata() -> Dict[str, Any]:
    """Return Git commit hash and dirty status.

    Returns an empty dict if Git is unavailable or the repo has no commits.
    """
    meta: Dict[str, Any] = {}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            meta["commit_hash"] = result.stdout.strip()
        result_dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if result_dirty.returncode == 0:
            meta["dirty"] = bool(result_dirty.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_run_directory(
    experiment_name: str,
    config: ExperimentConfig,
    run_id: Optional[str] = None,
    runs_root: Path = _RUNS_ROOT,
) -> Path:
    """Create a structured run directory and persist configuration.

    Parameters
    ----------
    experiment_name:
        Short experiment label (used as subdirectory).
    config:
        Resolved experiment configuration.
    run_id:
        Explicit run identifier.  If omitted, a timestamp-based ID is
        generated.
    runs_root:
        Root directory for all runs (default: ``runs/``).

    Returns
    -------
    Path to the created run directory.
    """
    run_id = run_id or config.run_id or _timestamp_run_id()
    config = dataclasses.replace(
        config,
        experiment_name=experiment_name,
        run_id=run_id,
    )

    run_dir = runs_root / experiment_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # Subdirectories
    (run_dir / "metrics").mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "probes").mkdir()

    # Config
    config.save(str(run_dir / "config.json"))

    # Environment metadata
    env_info = _gather_environment()
    with open(run_dir / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2, sort_keys=True)

    # Git metadata
    git_info = _gather_git_metadata()
    with open(run_dir / "git_metadata.json", "w") as f:
        json.dump(git_info, f, indent=2, sort_keys=True)

    return run_dir


def find_run_dir(
    experiment_name: str,
    run_id: str,
    runs_root: Path = _RUNS_ROOT,
) -> Optional[Path]:
    """Return the path to a run directory if it exists, else ``None``."""
    candidate = runs_root / experiment_name / run_id
    return candidate if candidate.is_dir() else None


def load_run_config(
    experiment_name: str,
    run_id: str,
    runs_root: Path = _RUNS_ROOT,
) -> Optional[ExperimentConfig]:
    """Load the ``config.json`` from a previous run."""
    run_dir = find_run_dir(experiment_name, run_id, runs_root)
    if run_dir is None:
        return None
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    return ExperimentConfig.load(str(config_path))
