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
    ACTIVE_SET_M_CANDIDATES,
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

    _write_results_md_from_disk(out_dir)
    print(f"Done. Results in {out_dir.resolve()}", file=sys.stderr)


def _write_results_md_from_disk(out_dir: Path) -> None:
    """Write RESULTS.md from saved JSON summaries."""
    md_dir = out_dir
    lines: list[str] = [
        "# 00 Corpus Inventory — RESULTS\n",
        "Data root: `data/rollouts/rwm_deterministic/scenario_0/`\n",
        "Seeds: 42, 43\n",
        "---\n",
    ]

    for seed_label in sorted(md_dir.glob("seed*")):
        json_path = seed_label / "corpus_summary.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            s = json.load(f)

        lines.append(f"## {seed_label.name}\n")
        lines.append(f"- Files: {s['n_files']} train, {s['n_val_files']} val")
        lines.append(f"- Transitions: {s['n_transitions']}")
        lines.append(f"- Eligible pointers: {s['n_eligible_pointers']}")
        lines.append(f"- Disjoint train/val: {s['train_val_disjoint']}")
        lines.append(f"- Selected H: {s['selected_H']}, selected h: {s['selected_h']}")
        lines.append(f"- Selected crowding rho: {s['selected_crowding_rho']}")
        lines.append(f"- Any legacy_done: {s['any_legacy_done']}")
        lines.append(f"- Quantize decimals: {s['quantize_decimals']}")
        lines.append(f"- Ep lengths: {s['episode_length_summary']}")
        lines.append(f"- Immediate reward: {s['immediate_reward_summary']}")
        lines.append("")

        lines.append("### Return quantiles\n")
        for H_label, qdict in sorted(s["return_quantiles"].items()):
            lines.append(f"- {H_label}: median={qdict['median']}, mean={qdict['mean']}, std={qdict['std']}")
        lines.append("")

        lines.append("### Surprise counts\n")
        for h_label, sc in sorted(s["surprise_counts"].items()):
            lines.append(f"- {h_label}: up={sc['up_count']} ({sc['up_pct']}%), down={sc['down_count']} ({sc['down_pct']}%)")
        lines.append("")

        lines.append("### Sensitivity grid (ESS)\n")
        lines.append(f"| Config | ESS | ESS/N | selected_H | selected_h |")
        lines.append(f"|--------|-----|-------|------------|------------|")
        for cfg in s["sensitivity_grid"]:
            lines.append(
                f"| {cfg['name']} | {cfg['effective_sample_size']} | "
                f"{cfg['ess_ratio']} | {cfg['selected_H']} | {cfg['selected_h']} |"
            )
        lines.append("")

        lines.append("### Active-set simulation\n")
        as_data = s["sensitivity_grid"][0].get("active_set_simulation", {})
        for M_label, sim in sorted(as_data.items()):
            if isinstance(sim, dict) and "skipped" in sim:
                continue
            lines.append(f"- M={M_label}: cycles={sim.get('n_cycles', '?')}, "
                         f"replacement={sim.get('mean_replacement_fraction', '?')}, "
                         f"jaccard_distance={sim.get('mean_jaccard_distance', '?')}, "
                         f"unique_pct={sim.get('unique_pointers_pct', '?')}%")
        lines.append("")

        lines.append("### Density metrics\n")
        dm = s.get("density_metrics", {})
        if dm:
            lines.append(f"- Highest-weight 10% mass: {dm.get('highest_weight_10pct_mass_fraction')}")
            lines.append(f"- Lowest-return 10% weight: {dm.get('lowest_return_10pct_weight_share')}")
            lines.append(f"- Highest-return 10% weight: {dm.get('highest_return_10pct_weight_share')}")
            lines.append(f"- Largest equal-return group: {dm.get('largest_equal_return_group')}")
            lines.append(f"- Gini weight: {dm.get('gini_weight')}")
        lines.append("")

        lines.append("### Equal-return crowding sensitivity\n")
        lines.append("| rho | ESS/N | largest-group weight |")
        lines.append("|-----|-------|----------------------|")
        for rho_label, row in sorted(
            s["crowding_sensitivity"].items(), key=lambda item: float(item[0])
        ):
            group = row["density_metrics"]["largest_equal_return_group"]
            lines.append(
                f"| {rho_label} | {row['ess_ratio']} | {group['weight_share']} |"
            )
        lines.append("")
        lines.append("---\n")

    results_path = md_dir / "RESULTS.md"
    with open(results_path, "w") as f:
        f.write("\n".join(lines) + "\n")


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
