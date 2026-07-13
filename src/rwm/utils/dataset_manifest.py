"""Dataset manifest for reproducible data provenance.

A manifest records the exact rollout files, their content hashes, the
episode-safe train/validation partition, and preprocessing settings used
by a run.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from rwm.data.rollout_dataset import episode_safe_train_val_split

_MANIFEST_SCHEMA_VERSION = 1


def _file_hash(path: Path, chunk_bytes: int = 2**16) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _rel_paths(files: List[Path], root: Path) -> List[str]:
    """Convert absolute paths to relative (under *root*)."""
    return [str(f.relative_to(root)) for f in files]


def _abs_paths(rel_paths: List[str], root: Path) -> List[Path]:
    """Convert relative paths back to absolute."""
    return [root / p for p in rel_paths]


def build_dataset_manifest(
    data_root: Path,
    sequence_len: int = 16,
    image_size: int = 64,
    val_ratio: float = 0.2,
    shuffle_seed: int = 42,
    store_hashes: bool = True,
    include_done: bool = False,
) -> Dict:
    """Build a dataset manifest from a rollout data directory.

    Parameters
    ----------
    data_root:
        Root directory containing ``.npz`` rollout files (searched
        recursively).
    sequence_len, image_size:
        Preprocessing settings stored for provenance.
    val_ratio, shuffle_seed:
        Episode-safe split parameters.
    store_hashes:
        If ``True``, compute SHA-256 hashes for every file.
    include_done:
        Whether done windows were included.

    Returns
    -------
    Manifest dict ready for JSON serialization.

    Raises
    ------
    ValueError
        If fewer than 2 rollout files are found.
    """
    train_files, val_files = episode_safe_train_val_split(
        data_root, val_ratio=val_ratio, shuffle_seed=shuffle_seed,
    )

    all_files = sorted(train_files + val_files, key=str)
    train_set = set(train_files)
    val_set = set(val_files)

    assert train_set.isdisjoint(val_set), "Train/val files overlap"

    file_entries: List[Dict] = []
    for f in all_files:
        entry: Dict = {
            "path": str(f.relative_to(data_root)),
            "size_bytes": f.stat().st_size,
        }
        if store_hashes:
            entry["sha256"] = _file_hash(f)
        file_entries.append(entry)

    manifest: Dict = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "data_root": str(data_root.resolve()),
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "num_files": len(all_files),
        "sequence_len": sequence_len,
        "image_size": image_size,
        "include_done": include_done,
        "split": {
            "method": "episode_safe",
            "val_ratio": val_ratio,
            "shuffle_seed": shuffle_seed,
            "num_train": len(train_files),
            "num_val": len(val_files),
            "train_files": sorted(
                str(f.relative_to(data_root)) for f in train_files
            ),
            "val_files": sorted(
                str(f.relative_to(data_root)) for f in val_files
            ),
        },
        "files": file_entries,
    }
    return manifest


def validate_manifest(
    manifest: Dict,
    data_root: Optional[Path] = None,
) -> List[str]:
    """Validate a manifest, returning a list of issues (empty = valid).

    Checks:
    - Schema version matches.
    - Referenced files exist.
    - Referenced file hashes match when the manifest recorded them.
    - Train and val file lists are disjoint.
    """
    issues: List[str] = []

    if manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
        issues.append(
            f"Unsupported schema version {manifest.get('schema_version')} "
            f"(expected {_MANIFEST_SCHEMA_VERSION})"
        )

    if data_root is None:
        data_root = Path(manifest.get("data_root", "."))
    else:
        data_root = Path(data_root)

    # Check referenced files exist
    for entry in manifest.get("files", []):
        fpath = data_root / entry["path"]
        if not fpath.exists():
            issues.append(f"Missing file: {fpath}")
        elif "sha256" in entry and _file_hash(fpath) != entry["sha256"]:
            issues.append(f"Hash mismatch: {fpath}")

    # Check disjoint train/val
    split = manifest.get("split", {})
    train_set = set(split.get("train_files", []))
    val_set = set(split.get("val_files", []))
    overlap = train_set & val_set
    if overlap:
        issues.append(
            f"Train/val overlap: {len(overlap)} files in both sets"
        )

    # Check counts match
    if len(train_set) + len(val_set) != manifest.get("num_files", 0):
        issues.append(
            "train_files + val_files != num_files"
        )

    return issues


def save_manifest(manifest: Dict, path: Path) -> None:
    """Save a manifest to a JSON file."""
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def load_manifest(path: Path) -> Dict:
    """Load a manifest from a JSON file."""
    with open(path) as f:
        return json.load(f)
