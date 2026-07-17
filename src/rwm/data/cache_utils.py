"""Shared frame-cache utility — key derivation, validation, manifest loading.

Cache location: ``data/cache/rollout_frames_v1/``
Manifest: ``manifest.json`` with schema version, image size, transform
spec, and ``file_map`` (relative source path → cache key).

Key derivation: SHA-256(content_hash + schema_version + image_size + transform_spec)
A different image size, transform, or source content produces a different key.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_CACHE_SCHEMA_VERSION = 1
_TRANSFORM_SPEC = "ToTensor+Resize"
_DEFAULT_IMAGE_SIZE = 64
_PROJECT_CACHE_DIR = Path("data/cache/rollout_frames_v1")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def cache_key(source_path: Path, image_size: int = _DEFAULT_IMAGE_SIZE) -> str:
    """Deterministic cache key for one rollout file."""
    sha = _file_sha256(source_path)
    raw = f"{_CACHE_SCHEMA_VERSION}_{sha}_{image_size}_{_TRANSFORM_SPEC}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_manifest(cache_dir: Path) -> dict:
    """Load and validate the cache manifest.  Raises ``ValueError`` on
    schema/image-size/transform mismatch."""
    man_path = cache_dir / "manifest.json"
    if not man_path.exists():
        raise ValueError(
            f"Cache manifest not found at {man_path}. "
            "Build the cache first:\n"
            f"  python scripts/build_frame_cache.py --cache-dir {cache_dir}"
        )
    with open(man_path) as f:
        manifest = json.load(f)

    sv = manifest.get("schema_version")
    if sv != _CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Cache schema version mismatch: expected {_CACHE_SCHEMA_VERSION}, got {sv}. "
            "Rebuild the cache."
        )
    im = manifest.get("image_size", 0)
    if im != _DEFAULT_IMAGE_SIZE:
        raise ValueError(
            f"Cache image size mismatch: expected {_DEFAULT_IMAGE_SIZE}, got {im}."
        )
    return manifest


def _lookup_key(
    source_path: Path,
    manifest_file_map: Dict[str, str],
    data_root: Optional[Path] = None,
) -> Optional[str]:
    """Look up the cache key for *source_path* in *manifest_file_map*.

    Tries:
    1. Relative to ``data_root`` if provided.
    2. Just the filename.
    3. Last 2–5 path components as fallback.
    """
    if data_root is not None:
        try:
            rel = str(source_path.relative_to(data_root))
            if rel in manifest_file_map:
                return manifest_file_map[rel]
        except ValueError:
            pass

    # Try filename only
    if source_path.name in manifest_file_map:
        return manifest_file_map[source_path.name]

    # Try progressively longer suffix matches
    parts = source_path.parts
    for n_parts in range(2, min(len(parts), 6) + 1):
        candidate = str(Path(*parts[-n_parts:]))
        if candidate in manifest_file_map:
            return manifest_file_map[candidate]

    return None


def verify_cache_entry(
    cache_dir: Path,
    source_path: Path,
    manifest_file_map: Dict[str, str],
    image_size: int = _DEFAULT_IMAGE_SIZE,
    data_root: Optional[Path] = None,
) -> Path:
    """Verify a cache entry exists and matches its source.

    Returns the cache file path on success.

    Raises ``ValueError`` with a descriptive message on any mismatch.
    """
    expected_key = _lookup_key(source_path, manifest_file_map, data_root)

    if expected_key is None:
        raise ValueError(
            f"Source {source_path.name} not found in cache manifest. "
            "Rebuild the cache:\n"
            f"  python scripts/build_frame_cache.py"
        )

    actual_key = cache_key(source_path, image_size)
    if actual_key != expected_key:
        raise ValueError(
            f"Cache key mismatch for {source_path.name}: "
            f"manifest has {expected_key}, source produces {actual_key}. "
            "The source file has changed since caching. Rebuild the cache:\n"
            f"  python scripts/build_frame_cache.py"
        )

    cache_path = cache_dir / f"{expected_key}.npy"
    if not cache_path.exists():
        raise ValueError(
            f"Cache file missing: {cache_path}. Rebuild the cache:\n"
            f"  python scripts/build_frame_cache.py"
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


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()
