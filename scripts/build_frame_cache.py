#!/usr/bin/env python3
"""Build a pre-processed frame cache for rollout files.

Transforms each episode's uint8 observations to (T, 3, 64, 64) float32
tensors once, saving repeated NPZ decompression and PIL resize in training.

Usage:
    python scripts/build_frame_cache.py                         # build default cache
    python scripts/build_frame_cache.py --cache-dir /custom/path
    python scripts/build_frame_cache.py --dry-run               # report what would be cached
    python scripts/build_frame_cache.py --validate              # check integrity
    python scripts/build_frame_cache.py --data-root data/rollouts/rwm_deterministic
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

_CACHE_SCHEMA_VERSION = 1


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _episode_hash_key(source_path: Path, image_size: int) -> str:
    """Deterministic cache key based on source content + preprocessing."""
    sha = _file_sha256(source_path)
    raw = f"{_CACHE_SCHEMA_VERSION}_{sha}_{image_size}_ToTensor_Resize"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_cache(
    data_root: Path,
    cache_dir: Path,
    image_size: int = 64,
    dry_run: bool = False,
) -> dict:
    """Build frame cache for all rollout .npz files under data_root.

    Returns summary dict.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size), antialias=True),
    ])

    npz_files = sorted(data_root.rglob("*.npz"))
    # Exclude evaluation files
    npz_files = [f for f in npz_files if not f.with_suffix(".episode.json").exists()
                 and not f.with_suffix(".branch.json").exists()]

    cache_dir.mkdir(parents=True, exist_ok=True)

    built = 0
    skipped = 0
    total_bytes = 0
    start = time.time()

    for src in npz_files:
        key = _episode_hash_key(src, image_size)
        cache_path = cache_dir / f"{key}.npy"

        if cache_path.exists():
            skipped += 1
            total_bytes += cache_path.stat().st_size
            continue

        if dry_run:
            print(f"Would cache: {src} -> {cache_path.name}")
            skipped += 1
            continue

        data = np.load(src)
        raw_obs = data["obs"]  # (T, H, W, C) uint8
        T = raw_obs.shape[0]

        # Transform each frame
        frames = []
        for t in range(T):
            img = Image.fromarray(raw_obs[t])
            tensor = transform(img)  # (3, H, W) float32 [0,1]
            frames.append(tensor.numpy())

        stacked = np.stack(frames, axis=0).astype(np.float32)  # (T, 3, H, W)

        # Validate shape
        assert stacked.shape == (T, 3, image_size, image_size), f"Shape mismatch: {stacked.shape}"

        np.save(cache_path, stacked)
        total_bytes += cache_path.stat().st_size
        built += 1

        if built % 5 == 0:
            print(f"  Cached {built}/{len(npz_files)} ({src.name})")

    elapsed = time.time() - start
    summary = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "cache_dir": str(cache_dir.resolve()),
        "data_root": str(data_root.resolve()),
        "image_size": image_size,
        "total_files": len(npz_files),
        "built": built,
        "skipped": skipped,
        "total_mb": round(total_bytes / 1e6, 2),
        "elapsed_s": round(elapsed, 1),
    }

    # Write manifest
    manifest = summary.copy()
    manifest["file_map"] = {
        str(src.relative_to(data_root)): _episode_hash_key(src, image_size)
        for src in npz_files
    }
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return summary


def validate_cache(cache_dir: Path, data_root: Path, image_size: int = 64) -> list:
    """Check that every cached file's key matches its source.

    Returns list of issue strings (empty = valid).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size), antialias=True),
    ])
    issues = []

    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        issues.append("manifest.json not found")
        return issues

    with open(manifest_path) as f:
        manifest = json.load(f)

    for rel_path, expected_key in manifest.get("file_map", {}).items():
        src = data_root / rel_path
        if not src.exists():
            issues.append(f"Source missing: {src}")
            continue
        actual_key = _episode_hash_key(src, image_size)
        if actual_key != expected_key:
            issues.append(f"Key mismatch for {rel_path}: expected {expected_key}, got {actual_key}")
        cache_path = cache_dir / f"{expected_key}.npy"
        if not cache_path.exists():
            issues.append(f"Cache missing: {cache_path}")
            continue
        # Verify shape
        arr = np.load(cache_path)
        expected_shape = (arr.shape[0], 3, image_size, image_size)
        if arr.shape != (arr.shape[0], 3, image_size, image_size):
            issues.append(f"Shape mismatch for {cache_path.name}: {arr.shape}")

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path("data/rollouts/rwm_deterministic/scenario_0"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("data/cache/rollout_frames_v1"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    if args.validate:
        issues = validate_cache(args.cache_dir, args.data_root, args.image_size)
        if issues:
            print("Validation issues:")
            for i in issues:
                print(f"  - {i}")
        else:
            print("Cache valid: all entries match source hashes and shapes.")
        return

    summary = build_cache(args.data_root, args.cache_dir, args.image_size, args.dry_run)
    print(f"\nCache build summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
