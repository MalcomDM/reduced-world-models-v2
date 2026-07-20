#!/usr/bin/env python3
"""Stage 7.0A — Factual Memory Corpus Inventory CLI.

Usage:
    python scripts/diagnostics/corpus_inventory.py \
        --data-root data/rollouts/rwm_deterministic/scenario_0/ \
        --seeds 42 43 \
        --out runs/component_refinement/memory/00_corpus_inventory

Read-only. No models, trainers, caches, or samplers are created.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from rwm.memory.corpus_profiler import (
    profile_corpus,
    DEFAULT_HORIZONS,
    DEFAULT_D_HORIZONS,
    SENSITIVITY_GRID,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 7.0A — Factual Memory Corpus Inventory",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/rollouts/rwm_deterministic/scenario_0/",
        help="Root directory containing .npz rollout files",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43],
        help="Data split seeds to profile",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="runs/component_refinement/memory/00_corpus_inventory",
        help="Output directory for JSON results",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Factual return horizons",
    )
    parser.add_argument(
        "--d-horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_D_HORIZONS),
        help="Directional change windows",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: dict[str, dict] = {}
    for seed in args.seeds:
        print(f"Profiling seed {seed}...", file=sys.stderr)
        summary = profile_corpus(
            data_root=data_root,
            data_split_seed=seed,
            horizons=tuple(args.horizons),
            d_horizons=tuple(args.d_horizons),
        )
        seed_dir = out_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        out_path = seed_dir / "corpus_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=_json_fallback)
        print(f"  Wrote {out_path}", file=sys.stderr)
        all_summaries[f"seed{seed}"] = summary

    _write_results_md(all_summaries, out_dir, args)
    _write_run_index(out_dir)
    print(f"Done. Results in {out_dir.resolve()}", file=sys.stderr)


def _write_results_md(
    summaries: dict[str, dict],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Write a concise RESULTS.md with key numbers from both seeds."""
    lines = [
        "# 00 Corpus Inventory — RESULTS\n",
        f"Date: (generated)\n",
        f"Data root: `{args.data_root}`\n",
        f"Seeds: {args.seeds}\n",
        f"Horizons: {args.horizons}\n",
        f"d-horizons: {args.d_horizons}\n",
        "---\n",
    ]
    for seed_label in sorted(summaries.keys()):
        s = summaries[seed_label]
        lines.append(f"## {seed_label}\n")
        lines.append(f"- Files: {s['n_files']} train, {s['n_val_files']} val")
        lines.append(f"- Transitions: {s['n_transitions']}")
        lines.append(f"- Eligible pointers: {s['n_eligible_pointers']}")
        lines.append(f"- Disjoint train/val: {s['train_val_disjoint']}")
        lines.append(f"- Ep lengths: {s['episode_length_summary']}")
        lines.append(f"- Immediate reward: {s['immediate_reward_summary']}")
        lines.append("")
        lines.append("### Return quantiles\n")
        for H_label, qdict in sorted(s["return_quantiles"].items()):
            lines.append(f"- {H_label}: {qdict}")
        lines.append("")
        lines.append("### Surprise counts\n")
        for h_label, sc in sorted(s["surprise_counts"].items()):
            lines.append(f"- {h_label}: {sc}")
        lines.append("")
        lines.append("### Sensitivity grid (ESS)\n")
        lines.append("| Config | ESS | ESS ratio |")
        lines.append("|--------|-----|-----------|")
        for cfg in s["sensitivity_grid"]:
            lines.append(
                f"| {cfg['name']} | {cfg['effective_sample_size']} | "
                f"{cfg['ess_ratio']} |"
            )
        lines.append("")
        lines.append("### Dense-region impact\n")
        lines.append(f"- {s['dense_region_impact']}")
        lines.append("")
        lines.append("---\n")
    lines.append("")
    results_path = out_dir / "RESULTS.md"
    with open(results_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_run_index(out_dir: Path) -> None:
    """Write RUN_INDEX.md if not present."""
    index_path = out_dir.parent / "RUN_INDEX.md"
    if index_path.exists():
        return
    content = """# Memory Component RUN_INDEX

| Directory | Status | Description |
|-----------|--------|-------------|
| 00_corpus_inventory | DONE | Factual corpus profiler; seeds 42 & 43 profiled |
| 01_uniform_replay | — | Reserved for uniform dream baseline |
| 02_probabilistic_replay | — | Reserved for probabilistic priority replay |
| 03_latent_cache_ablation | — | Reserved for cached-z vs reconstruction |
| 04_wake_dream_cycle | — | Reserved for first wake-dream refresh |
"""
    with open(index_path, "w") as f:
        f.write(content)


def _json_fallback(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


if __name__ == "__main__":
    main()
