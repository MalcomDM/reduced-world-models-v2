"""Shared frame-cache utility — key derivation, validation, manifest loading.

Cache location: ``data/cache/rollout_frames_v1/``
Manifest: ``manifest.json`` with schema version, image size, transform
spec, ``data_root``, and ``file_map`` (source_path_relative_to_data_root → key).

Key derivation: SHA-256(content_hash + schema_version + image_size + transform_spec)
A different image size, transform, or source content produces a different key.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_CACHE_SCHEMA_VERSION = 1
_TRANSFORM_SPEC = "ToTensor+Resize"
_DEFAULT_IMAGE_SIZE = 64


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

def cache_key(source_path: Path, image_size: int = _DEFAULT_IMAGE_SIZE) -> str:
    """Deterministic cache key for one rollout file."""
    sha = _file_sha256(source_path)
    raw = f"{_CACHE_SCHEMA_VERSION}_{sha}_{image_size}_{_TRANSFORM_SPEC}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Manifest load / validate
# ---------------------------------------------------------------------------

def load_manifest(
    cache_dir: Path,
    image_size: int = _DEFAULT_IMAGE_SIZE,
    transform_spec: str = _TRANSFORM_SPEC,
) -> dict:
    """Load and validate the cache manifest.

    Checks schema version, image size, and transform specification.
    Raises ``ValueError`` on any mismatch.
    """
    man_path = cache_dir / "manifest.json"
    if not man_path.exists():
        raise ValueError(
            f"Cache manifest not found at {man_path}. "
            "Build the cache first:\n"
            f"  python scripts/data/build_frame_cache.py --cache-dir {cache_dir}"
        )
    with open(man_path) as f:
        manifest = json.load(f)

    sv = manifest.get("schema_version")
    if sv != _CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Cache schema version mismatch: expected {_CACHE_SCHEMA_VERSION}, "
            f"got {sv}. Rebuild the cache."
        )
    im = manifest.get("image_size", 0)
    if im != image_size:
        raise ValueError(
            f"Cache image size mismatch: requested {image_size}, cache has {im}. "
            "Rebuild the cache with the correct image size."
        )
    ts = manifest.get("transform_spec", "")
    if ts != transform_spec:
        raise ValueError(
            f"Cache transform mismatch: requested {transform_spec!r}, "
            f"cache has {ts!r}. A custom transform cannot use a cache built "
            "for the default transform."
        )
    return manifest


# ---------------------------------------------------------------------------
# Entry verification (exact data-root lookup only)
# ---------------------------------------------------------------------------

def verify_cache_entry(
    cache_dir: Path,
    source_path: Path,
    manifest: dict,
    image_size: int = _DEFAULT_IMAGE_SIZE,
) -> Path:
    """Verify a cache entry via exact ``data_root``-based lookup.

    ``source_path`` must be under ``manifest['data_root']``.  The relative
    path is looked up in ``manifest['file_map']``.  No fuzzy/suffix matching
    is performed.

    Returns the cache file path on success.

    Raises ``ValueError`` with a descriptive message on any mismatch.
    """
    data_root_str = manifest.get("data_root")
    if not data_root_str:
        raise ValueError("Cache manifest is missing data_root. Rebuild the cache.")
    data_root = Path(data_root_str).resolve()

    try:
        rel = str(source_path.resolve().relative_to(data_root))
    except ValueError:
        raise ValueError(
            f"Source {source_path} is not under cache data_root {data_root}. "
            "Cannot use this cache for this file."
        )

    file_map = manifest.get("file_map", {})
    expected_key = file_map.get(rel)
    if expected_key is None:
        raise ValueError(
            f"Source {rel} not found in cache manifest file_map. "
            "Rebuild the cache:\n"
            f"  python scripts/data/build_frame_cache.py"
        )

    actual_key = cache_key(source_path, image_size)
    if actual_key != expected_key:
        raise ValueError(
            f"Cache key mismatch for {rel}: "
            f"manifest has {expected_key}, source produces {actual_key}. "
            "The source file has changed since caching. Rebuild the cache:\n"
            f"  python scripts/data/build_frame_cache.py"
        )

    cache_path = cache_dir / f"{expected_key}.npy"
    if not cache_path.exists():
        raise ValueError(
            f"Cache file missing: {cache_path}. Rebuild the cache:\n"
            f"  python scripts/data/build_frame_cache.py"
        )

    # Validate shape
    arr = np.load(cache_path, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(
            f"Cache entry {cache_path.name}: expected 4D (T, C, H, W), "
            f"got {arr.ndim}D. Rebuild the cache."
        )
    if arr.shape[1] != 3 or arr.shape[2] != image_size or arr.shape[3] != image_size:
        raise ValueError(
            f"Cache entry {cache_path.name}: expected (T, 3, {image_size}, {image_size}), "
            f"got {arr.shape}. Rebuild the cache."
        )

    return cache_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()
