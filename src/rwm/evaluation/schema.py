"""Versioned evaluation rollout schema and seed manifest types.

Schema version 1 (new evaluation data):
  - Separate terminated/truncated (no longer merged into done).
  - Purpose must be 'evaluation_only'.
  - Includes environment seed, policy provenance, git config reference.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

_EVAL_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Split(str, Enum):
    DEV = "dev"
    VAL = "val"
    LOCKED_TEST = "locked_test"


class Quality(str, Enum):
    UNREVIEWED = "unreviewed"
    REVIEW = "review"
    KEEP = "keep"
    DISCARD = "discard"


# ---------------------------------------------------------------------------
# Episode metadata (saved as sidecar .json)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class EpisodeMetadata:
    """Metadata for one evaluation episode.

    Saved alongside the ``.npz`` rollout file as ``{filename}.episode.json``.
    """
    schema_version: int = _EVAL_SCHEMA_VERSION
    purpose: str = "evaluation_only"
    split: str = ""

    episode_id: str = ""
    env_id: str = "CarRacing-v3"
    env_version: str = ""
    track_seed: int = 0
    policy: str = ""

    collector_timestamp: str = ""
    git_commit: str = ""
    config_ref: str = ""

    # Collector settings (serialized as JSON string)
    max_steps: int = 1000
    early_push: int = 0
    idle_threshold: int = 100
    render_mode: str = "rgb_array"

    # Manifest provenance
    manifest_hash: str = ""
    manifest_path: str = ""

    terminated: bool = False
    truncated: bool = False
    steps: int = 0

    quality: str = Quality.UNREVIEWED.value
    scenario_tags: str = ""
    operator: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Seed manifest
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SeedManifest:
    """Immutable mapping of seeds to splits.

    Each seed belongs to exactly one split.  A seed manifest must exist
    before any evaluation rollout can be collected.
    """
    schema_version: int = _EVAL_SCHEMA_VERSION
    created_at: str = ""
    entries: Dict[str, str] = dataclasses.field(default_factory=dict)
    # entry: seed_str -> split_name

    @staticmethod
    def validate_entries(entries: Dict[str, str]) -> List[str]:
        """Validate seed entries WITHOUT mutating state.
        Use this BEFORE constructing a manifest to reject duplicates early.
        """
        issues: List[str] = []
        seen: Dict[str, str] = {}
        for seed_str, split in entries.items():
            try:
                int(seed_str)
            except ValueError:
                issues.append(f"Invalid seed (not an integer): {seed_str!r}")
                continue
            if split not in (Split.DEV.value, Split.VAL.value, Split.LOCKED_TEST.value):
                issues.append(f"Invalid split for seed {seed_str}: {split!r}")
            if seed_str in seen:
                issues.append(
                    f"Duplicate seed {seed_str}: split {seen[seed_str]} and {split}"
                )
            seen[seed_str] = split
        if not entries:
            issues.append("Seed manifest is empty")
        return issues

    def validate(self) -> List[str]:
        return self.validate_entries(self.entries)

    def assert_valid(self) -> None:
        issues = self.validate()
        if issues:
            raise ValueError("Seed manifest issues:\n" + "\n".join(issues))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_episode_id(track_seed: int, split: str, timestamp: str) -> str:
    raw = f"{split}_{track_seed}_{timestamp}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def make_episode_metadata(
    track_seed: int,
    split: str,
    policy: str = "",
    git_commit: str = "",
    config_ref: str = "",
) -> EpisodeMetadata:
    now = datetime.datetime.utcnow().isoformat() + "Z"
    ep_id = _new_episode_id(track_seed, split, now)
    return EpisodeMetadata(
        episode_id=ep_id,
        split=split,
        track_seed=track_seed,
        policy=policy,
        collector_timestamp=now,
        git_commit=git_commit,
        config_ref=config_ref,
    )


def save_episode_metadata(meta: EpisodeMetadata, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(meta), f, indent=2, sort_keys=True)


def load_episode_metadata(path: Path) -> EpisodeMetadata:
    with open(path) as f:
        data = json.load(f)
    return EpisodeMetadata(**data)


def _compute_manifest_hash(manifest_path: Path) -> str:
    """SHA-256 hex digest (first 16 chars) of a seed manifest file."""
    import hashlib
    h = hashlib.sha256()
    h.update(manifest_path.read_bytes())
    return h.hexdigest()[:16]


def validate_episode_integrity(
    meta: EpisodeMetadata,
    manifest: SeedManifest,
    manifest_path: Path,
    expected_split: Optional[str] = None,
) -> List[str]:
    """Return provenance violations for one standard evaluation episode.

    This is deliberately shared by the status command and evaluation code so
    a status warning cannot be accidentally ignored when reporting metrics.
    """
    issues: List[str] = []
    manifest.assert_valid()
    if meta.purpose != "evaluation_only":
        issues.append(f"purpose must be evaluation_only, got {meta.purpose!r}")
    if not meta.manifest_hash:
        issues.append("missing manifest hash")
    elif meta.manifest_hash != _compute_manifest_hash(manifest_path):
        issues.append("manifest hash mismatch")
    if not meta.manifest_path:
        issues.append("missing manifest path")
    elif Path(meta.manifest_path).resolve() != manifest_path.resolve():
        issues.append("manifest path mismatch")

    declared_split = manifest.entries.get(str(meta.track_seed))
    if declared_split is None:
        issues.append(f"seed {meta.track_seed} is absent from manifest")
    elif meta.split != declared_split:
        issues.append(
            f"metadata split {meta.split!r} != manifest split {declared_split!r}"
        )
    if expected_split is not None and meta.split != expected_split:
        issues.append(f"metadata split {meta.split!r} != directory split {expected_split!r}")
    return issues


def save_seed_manifest(manifest: SeedManifest, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(manifest), f, indent=2, sort_keys=True)


def load_seed_manifest(path: Path) -> SeedManifest:
    with open(path) as f:
        data = json.load(f)
    return SeedManifest(**data)
