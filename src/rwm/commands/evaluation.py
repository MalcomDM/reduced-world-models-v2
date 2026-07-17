"""CLI commands for evaluation data management.

Commands:
  eval-init-seeds     Initialize a seed manifest from template; refuses overwrite.
  eval-collect        Collect one evaluation episode on a registered seed.
  eval-label          Label a saved episode preserving all provenance.
  eval-status         Show split counts, quality breakdown, integrity issues.
"""

import dataclasses
import datetime
import json
from pathlib import Path
from typing import Optional

import typer
import numpy as np

from rwm.evaluation.schema import (
    SeedManifest,
    Split,
    Quality,
    EpisodeMetadata,
    load_seed_manifest,
    save_seed_manifest,
    load_episode_metadata,
    save_episode_metadata,
    _compute_manifest_hash,
    validate_episode_integrity,
)
from rwm.evaluation.collector import collect_evaluation_episode


app = typer.Typer()


# ---------------------------------------------------------------------------
# eval init-seeds
# ---------------------------------------------------------------------------

@app.command()
def init_seeds(
    output: Path = typer.Argument(..., help="Output path for seed manifest JSON"),
    dev_seeds: str = typer.Option("", help="Comma-separated development seeds"),
    val_seeds: str = typer.Option("", help="Comma-separated validation seeds"),
    test_seeds: str = typer.Option("", help="Comma-separated locked-test seeds"),
    force_replace: bool = typer.Option(False, "--force-replace",
        help="Overwrite an existing manifest (refused if episodes reference its hash)"),
):
    """Initialize a seed manifest from explicitly provided seed lists.

    Refuses to overwrite an existing manifest by default.
    """
    if output.exists():
        _check_manifest_collected(output)
        if not force_replace:
            typer.echo(
                f"Manifest exists at {output}. Use --force-replace to overwrite "
                "(only allowed if no episodes reference its hash).",
                err=True,
            )
            raise typer.Exit(1)

    entries: dict[str, str] = {}
    all_pairs: list[tuple[str, str]] = []

    def _add(s_str: str, split_val: str) -> None:
        for part in s_str.split(","):
            s = part.strip()
            if s:
                all_pairs.append((s, split_val))

    _add(dev_seeds, Split.DEV.value)
    _add(val_seeds, Split.VAL.value)
    _add(test_seeds, Split.LOCKED_TEST.value)

    seen: dict[str, str] = {}
    dup_issues: list[str] = []
    for s, split_val in all_pairs:
        if s in seen:
            dup_issues.append(f"Duplicate seed {s}: split {seen[s]} and {split_val}")
        seen[s] = split_val

    if dup_issues:
        typer.echo("Duplicate seed errors:", err=True)
        for i in dup_issues:
            typer.echo(f"  - {i}", err=True)
        raise typer.Exit(1)

    for s, split_val in all_pairs:
        entries[s] = split_val

    issues = SeedManifest.validate_entries(entries)
    if issues:
        typer.echo("Validation errors:", err=True)
        for i in issues:
            typer.echo(f"  - {i}", err=True)
        raise typer.Exit(1)

    manifest_obj = SeedManifest(
        created_at=datetime.datetime.utcnow().isoformat() + "Z",
        entries=entries,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_seed_manifest(manifest_obj, output)
    typer.echo(f"Seed manifest saved to {output}")
    typer.echo(f"  Dev: {dev_seeds or '(none)'}")
    typer.echo(f"  Val: {val_seeds or '(none)'}")
    typer.echo(f"  Test: {test_seeds or '(none)'}")


def _check_manifest_collected(manifest_path: Path) -> None:
    """Check if any episodes reference this manifest's hash."""
    manifest = load_seed_manifest(manifest_path)
    current_hash = _compute_manifest_hash(manifest_path)
    # Look for episodes in the same directory tree
    for split_name in ("dev", "val", "locked_test"):
        split_dir = manifest_path.parent / split_name
        if not split_dir.exists():
            continue
        for meta_path in split_dir.glob("*.episode.json"):
            meta = load_episode_metadata(meta_path)
            if meta.manifest_hash and meta.manifest_path:
                if meta.manifest_hash == current_hash:
                    typer.echo(
                        f"ERROR: Episode {meta.episode_id} references this manifest hash. "
                        "Cannot replace.", err=True,
                    )
                    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# eval collect
# ---------------------------------------------------------------------------

@app.command()
def collect(
    manifest: Path = typer.Argument(..., exists=True, help="Seed manifest JSON"),
    seed: int = typer.Argument(..., help="Track seed (must be in manifest)"),
    out_dir: Path = typer.Option(Path("data/eval")),
    max_steps: int = typer.Option(1000),
    early_push: int = typer.Option(0),
    idle_threshold: int = typer.Option(100),
    operator: str = typer.Option(""),
    policy_name: str = typer.Option("random_smooth",
        help="random | random_smooth | human"),
    render_mode: str = typer.Option("rgb_array",
        help="rgb_array (headless) or human (requires display)"),
    fps: int = typer.Option(60, min=1, help="Frames per second (human mode only)"),
    env_version: str = typer.Option(""),
):
    """Collect one evaluation episode on a registered seed.

    Split is derived from manifest, never from user override.

    Manual human collection: --policy-name human --render-mode human.
    This requires a display and keyboard.
    """
    if policy_name == "human" and render_mode != "human":
        typer.echo(
            "ERROR: --policy-name human requires --render-mode human.",
            err=True,
        )
        raise typer.Exit(1)

    if policy_name == "human":
        try:
            from rwm.policies.human_policy import HumanPolicy
            import pygame
            policy = HumanPolicy()
        except ImportError as e:
            typer.echo(f"Cannot use HumanPolicy: {e}", err=True)
            raise typer.Exit(1)
        clock = pygame.time.Clock()
    else:
        from rwm.policies.random_policy import RandomPolicy
        from rwm.policies.base_policy import BasePolicy

        smooth = (policy_name == "random_smooth")
        class _EvalPolicy(BasePolicy):
            def __init__(self) -> None:
                self._inner = RandomPolicy(smooth=smooth)
            def act(self, obs):
                return self._inner.act(obs)
            def reset(self):
                self._inner.reset()
        policy = _EvalPolicy()
        clock = None

    try:
        path = collect_evaluation_episode(
            manifest_path=manifest, seed=seed, out_dir=out_dir,
            policy_fn=policy.act, policy_name=policy_name, max_steps=max_steps,
            early_push=early_push, idle_threshold=idle_threshold,
            operator=operator, render_mode=render_mode, env_version=env_version,
            clock=clock, fps=fps, running_check=getattr(policy, "is_running", None),
            on_env_ready=policy.reset if policy_name == "human" else None,
        )
        typer.echo(f"Saved episode: {path}")
    finally:
        if policy_name == "human":
            import pygame
            pygame.quit()


# ---------------------------------------------------------------------------
# eval label
# ---------------------------------------------------------------------------

@app.command()
def label(
    episode_path: Path = typer.Argument(..., help="Path to .npz or .episode.json"),
    quality: str = typer.Option("unreviewed", help="keep | review | discard | unreviewed"),
    tags: str = typer.Option("", help="Comma-separated scenario tags"),
    operator: str = typer.Option(""),
    notes: str = typer.Option(""),
):
    """Label a saved episode with quality, tags, operator, and notes.

    All collection/provenance fields are preserved exactly.
    """
    if quality not in (Quality.KEEP.value, Quality.REVIEW.value,
                       Quality.DISCARD.value, Quality.UNREVIEWED.value):
        typer.echo(f"Invalid quality: {quality!r}", err=True)
        raise typer.Exit(1)

    meta_path = episode_path.with_suffix(".episode.json")
    if not meta_path.exists():
        meta_path = episode_path
    if not meta_path.exists() or meta_path.suffix != ".json":
        typer.echo(f"Cannot find episode metadata at {meta_path}", err=True)
        raise typer.Exit(1)

    meta = load_episode_metadata(meta_path)
    updated = dataclasses.replace(
        meta,
        quality=quality,
        scenario_tags=tags,
        operator=operator,
        notes=notes,
    )
    save_episode_metadata(updated, meta_path)
    typer.echo(f"Label saved to {meta_path}")


# ---------------------------------------------------------------------------
# eval status
# ---------------------------------------------------------------------------

@app.command()
def status(
    eval_dir: Path = typer.Argument(..., help="Evaluation data root"),
    manifest_path: Optional[Path] = typer.Option(None, help="Seed manifest JSON"),
):
    """Show split counts, quality breakdown, integrity issues."""
    from collections import Counter

    # Standard episodes
    split_files: dict[str, list[Path]] = {}
    for split_name in ("dev", "val", "locked_test"):
        d = eval_dir / split_name
        if d.exists():
            split_files[split_name] = sorted(d.glob("*.npz"))
        else:
            split_files[split_name] = []

    # Branch experiments
    branch_dir = eval_dir / "branches"
    branch_files = sorted(branch_dir.glob("*.npz")) if branch_dir.exists() else []

    typer.echo("=== Evaluation Data Status ===")
    for split_name in ("dev", "val", "locked_test"):
        files = split_files[split_name]
        typer.echo(f"  {split_name}: {len(files)} episodes")
        quality_counts: Counter = Counter()
        for f in files:
            mp = f.with_suffix(".episode.json")
            if mp.exists():
                meta = load_episode_metadata(mp)
                quality_counts[meta.quality] += 1
            else:
                quality_counts["missing_metadata"] += 1
        typer.echo(f"    quality: {dict(quality_counts)}")

    if branch_files:
        typer.echo(f"  branches: {len(branch_files)} experiments (stored separately)")

    # Integrity checks
    if manifest_path and manifest_path.exists():
        manifest = load_seed_manifest(manifest_path)
        current_hash = _compute_manifest_hash(manifest_path)
        typer.echo(f"\n  Manifest: {len(manifest.entries)} seeds, hash={current_hash}")

        mismatches: list[str] = []
        for split_name in ("dev", "val", "locked_test"):
            for f in split_files[split_name]:
                mp = f.with_suffix(".episode.json")
                if not mp.exists():
                    mismatches.append(f"{f.name}: missing .episode.json")
                    continue
                meta = load_episode_metadata(mp)
                for issue in validate_episode_integrity(
                    meta, manifest, manifest_path, expected_split=split_name
                ):
                    mismatches.append(f"{meta.episode_id}: {issue}")

        if mismatches:
            typer.echo("\n  INTEGRITY ISSUES:")
            for m in mismatches:
                typer.echo(f"    - {m}")
        else:
            typer.echo("  No integrity issues detected.")

    total = sum(len(v) for v in split_files.values())
    typer.echo(f"\n  Total episodes: {total}")
    if total == 0:
        typer.echo("  WARNING: no evaluation episodes found")
