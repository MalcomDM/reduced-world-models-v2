"""Stage 7.0A — Factual Memory Corpus Inventory.

Read-only profiler over eligible training-file transitions.
Produces per-pointer metrics, corpus diagnostics, and a sensitivity grid
for continuous priority weights.  Core formula primitives are imported
from ``priority.py`` to guarantee exact numerical parity.

Does not create or modify any model, trainer, cache, or sampler.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from rwm.data.split import collect_and_split
from rwm.memory.priority import (
    QUANTIZE_DECIMALS,
    EPSILON,
    DEFAULT_SELECTED_H,
    DEFAULT_CROWDING_RHO,
    quantize,
    file_hash,
    compute_factual_returns,
    compute_directional_change,
    percentile_rank,
    positive_tail_percentile,
    negative_tail_percentile,
    compute_priority_score,
    apply_equal_return_crowding as _priority_crowding,
    compute_probabilities,
    effective_sample_size,
    reservoir_sample,
    _stable_uniforms,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HORIZONS: Tuple[int, ...] = (1, 2, 4, 8, 12)
DEFAULT_D_HORIZONS: Tuple[int, ...] = (2, 3, 4)
DEFAULT_SELECTED_D_H: int = 4

ACTIVE_SET_M_CANDIDATES: Tuple[int, ...] = (512, 1024, 2048)
N_CYCLE_SEEDS: int = 100
CROWDING_RHO_CANDIDATES: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)

# Sensitivity grid — interpretable candidate configurations
# Each entry: (name, eta, lambda_pos, lambda_neg, lambda_up, lambda_down,
#              lambda_legacy_done, alpha, beta)
SENSITIVITY_GRID: List[Tuple[str, float, float, float, float, float, float, float, float]] = [
    ("uniform",         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
    ("return_only",     0.1, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0),
    ("return_sharp",    0.1, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 1.0),
    ("change_focused",  0.1, 0.5, 0.5, 2.0, 2.0, 0.0, 1.0, 2.0),
    ("balanced",        0.1, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0),
    ("high_floor",      0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 1.0, 1.0),
    ("return_extreme",  0.1, 2.0, 2.0, 0.5, 0.5, 0.0, 2.0, 1.5),
]

# ---------------------------------------------------------------------------
# Compatibility wrappers — delegate to priority.py
# ---------------------------------------------------------------------------

# Re-export all formula primitives from priority.py so that existing
# profiler callers and tests continue to work unchanged.

# The canonical compute_weights signature differs (it takes an extra
# factual_returns argument for crowding).  We keep the profiler's
# grid-oriented signature that does NOT apply crowding internally
# (crowding is handled separately via apply_equal_return_crowding).


def compute_weights(
    q_pos: np.ndarray,
    q_neg: np.ndarray,
    q_up: np.ndarray,
    q_down: np.ndarray,
    legacy_done: np.ndarray,
    eta: float,
    lambda_pos: float,
    lambda_neg: float,
    lambda_up: float,
    lambda_down: float,
    lambda_legacy_done: float,
    alpha: float,
    beta: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Delegate to ``priority.compute_probabilities``.

    Computes raw priority score (no crowding) then mixes with uniform floor.
    """
    score = compute_priority_score(
        q_pos, q_neg, q_up, q_down, legacy_done,
        lambda_pos, lambda_neg, lambda_up, lambda_down,
        lambda_legacy_done, alpha, beta, eps,
    )
    return compute_probabilities(score, eta)


def apply_equal_return_crowding(
    weights: np.ndarray,
    factual_returns: np.ndarray,
    eta: float,
    rho: float,
) -> np.ndarray:
    """Delegate to ``priority.apply_equal_return_crowding``.

    Reconstructs the priority component from the mixture ``weights``,
    applies crowding, and re-mixes with the uniform floor.
    """
    n = len(weights)
    if n == 0:
        return weights.astype(np.float64, copy=True)
    floor_per_pointer = eta / n
    if eta == 1.0:
        return np.full(n, 1.0 / n, dtype=np.float64)
    priority = np.maximum(weights - floor_per_pointer, 0.0) / (1.0 - eta)
    crowded = _priority_crowding(priority, factual_returns, rho)
    if crowded.sum() <= 0:
        return np.full(n, 1.0 / n, dtype=np.float64)
    crowded /= crowded.sum()
    return floor_per_pointer + (1.0 - eta) * crowded


# ---------------------------------------------------------------------------
# Sensitivity grid runner
# ---------------------------------------------------------------------------


def run_sensitivity_grid(
    q_pos: np.ndarray,
    q_neg: np.ndarray,
    q_up: np.ndarray,
    q_down: np.ndarray,
    legacy_done: np.ndarray,
    n_pointers: int,
    selected_H: int,
    selected_h: int,
    file_names: Optional[List[str]] = None,
    pointer_ids: Optional[Sequence[str]] = None,
    q_ret: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Evaluate candidate weight configurations.

    Returns list of dicts with name, params, ESS, active-set simulation
    for M in ACTIVE_SET_M_CANDIDATES.
    """
    results = []
    for name, eta, lp, ln, lup, ldn, ld, alpha, beta in SENSITIVITY_GRID:
        w = compute_weights(
            q_pos, q_neg, q_up, q_down, legacy_done,
            eta, lp, ln, lup, ldn, ld, alpha, beta,
        )
        ess = effective_sample_size(w)
        finite_w = w[~np.isnan(w)]
        n_finite = len(finite_w)

        # Active-set simulation
        active_sets: Dict[str, Any] = {}
        for M in ACTIVE_SET_M_CANDIDATES:
            if (
                M > n_pointers
                or q_ret is None
                or file_names is None
                or pointer_ids is None
            ):
                active_sets[str(M)] = {"skipped": True}
                continue
            sim = simulate_active_set(
                w, M, q_ret, q_pos, q_neg, q_up, q_down,
                file_names, pointer_ids,
            )
            active_sets[str(M)] = sim

        results.append({
            "name": name,
            "params": {
                "eta": eta,
                "lambda_pos": lp,
                "lambda_neg": ln,
                "lambda_up": lup,
                "lambda_down": ldn,
                "lambda_legacy_done": ld,
                "alpha": alpha,
                "beta": beta,
            },
            "selected_H": selected_H,
            "selected_h": selected_h,
            "effective_sample_size": round(ess, 1),
            "ess_ratio": round(ess / max(1, n_finite), 4),
            "active_set_simulation": active_sets,
            "uniform_floor_contribution": round(eta, 4),
        })
    return results


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


def compute_signal_correlations(
    returns_by_H: Dict[int, np.ndarray],
    d_by_h: Dict[int, np.ndarray],
    q_pos: np.ndarray,
    q_neg: np.ndarray,
    q_up: Dict[int, np.ndarray],
    q_down: Dict[int, np.ndarray],
) -> Dict[str, float]:
    """Pairwise Pearson correlations among priority signals.

    Only non-NaN pairs are included. Returns a dict of ``"a_vs_b": r``.
    """
    corr: Dict[str, float] = {}
    signals: Dict[str, np.ndarray] = {}
    for H, arr in returns_by_H.items():
        signals[f"return_H={H}"] = arr
    for h, arr in d_by_h.items():
        signals[f"d_h={h}"] = arr
    signals["q_pos"] = q_pos
    signals["q_neg"] = q_neg
    for h, arr in q_up.items():
        signals[f"q_up_h={h}"] = arr
    for h, arr in q_down.items():
        signals[f"q_down_h={h}"] = arr
    keys = list(signals.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = signals[keys[i]]
            b = signals[keys[j]]
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() < 3:
                continue
            a_valid = a[mask]
            b_valid = b[mask]
            if np.std(a_valid) == 0.0 or np.std(b_valid) == 0.0:
                continue
            r = np.corrcoef(a_valid, b_valid)[0, 1]
            if not np.isnan(r):
                corr[f"{keys[i]}_vs_{keys[j]}"] = round(float(r), 4)
    return corr


# ---------------------------------------------------------------------------
# Episode contribution analysis
# ---------------------------------------------------------------------------


def compute_episode_contributions(
    file_names: List[str],
    q_ret: np.ndarray,
) -> Dict[str, Any]:
    """Contribution of each source episode across return quantile ranges.

    Returns dict mapping quantile range label -> {file: pointer_count}.
    """
    q_ranges = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (0.9, 1.0)]
    unique_files = list(dict.fromkeys(file_names))
    result: Dict[str, Any] = {}
    for lo, hi in q_ranges:
        label = f"q_{lo:.0%}_{hi:.0%}"
        mask = (q_ret >= lo) & (q_ret < hi) & ~np.isnan(q_ret)
        counts: Dict[str, int] = {}
        for idx in np.where(mask)[0]:
            fname = file_names[idx]
            counts[fname] = counts.get(fname, 0) + 1
        total = sum(counts.values())
        result[label] = {
            "total_pointers": int(total),
            "file_counts": {
                k: {"count": v, "pct": round(v / max(1, total) * 100, 1)}
                for k, v in sorted(counts.items(), key=lambda x: -x[1])
            },
        }
    return result


# ---------------------------------------------------------------------------
# Tie frequency analysis
# ---------------------------------------------------------------------------


def compute_tie_frequencies(values: np.ndarray) -> Dict[str, Any]:
    """Analyze tie frequency in a quantized value array.

    Returns count of unique values, number of tied groups >1, and the
    fraction of elements in ties.
    """
    finite = values[~np.isnan(values)]
    if len(finite) == 0:
        return {"unique_values": 0, "tied_groups": 0, "fraction_tied": 0.0}
    q = quantize(finite)
    unique, counts = np.unique(q, return_counts=True)
    tied_mask = counts > 1
    tied_elements = counts[tied_mask].sum()
    return {
        "unique_values": int(len(unique)),
        "tied_groups": int(tied_mask.sum()),
        "fraction_tied": round(float(tied_elements / len(finite)), 4),
        "largest_tie_size": int(counts.max()) if len(counts) > 0 else 1,
    }


# ---------------------------------------------------------------------------
# Dense-region / concentration metrics
# ---------------------------------------------------------------------------


def compute_density_metrics(
    weights: np.ndarray,
    q_ret: np.ndarray,
    file_names: List[str],
) -> Dict[str, Any]:
    """Corrected concentration and density metrics.

    Reports:
    - highest-weight 10% pointer mass
    - lowest-return 10% weight share
    - highest-return 10% weight share
    - largest quantized equal-return group: pointer share and weight share
    - weight Gini coefficient
    - per-episode maximum contribution
    """
    finite = ~(np.isnan(weights) | np.isnan(q_ret))
    w = weights[finite]
    q = q_ret[finite]
    fn_arr = np.array(file_names)[finite]
    N = len(w)
    if N == 0:
        return {}

    # Highest-weight 10%
    order_desc = np.argsort(w)[::-1]
    n10 = max(1, N // 10)
    top_w_idx = order_desc[:n10]
    top_w_mass = float(w[top_w_idx].sum() / w.sum())

    # Lowest-return 10% weight share
    order_q = np.argsort(q)
    low_ret_idx = order_q[:n10]
    low_ret_w_share = float(w[low_ret_idx].sum() / w.sum())

    # Highest-return 10% weight share
    high_ret_idx = order_q[-n10:]
    high_ret_w_share = float(w[high_ret_idx].sum() / w.sum())

    # Largest quantized equal-return group
    qq = quantize(q)
    unique_vals, counts = np.unique(qq, return_counts=True)
    largest_group_val = unique_vals[np.argmax(counts)]
    lg_mask = qq == largest_group_val
    lg_pointer_share = float(lg_mask.sum() / N)
    lg_weight_share = float(w[lg_mask].sum() / w.sum())

    # Gini
    gini = _gini(w)

    # Per-episode max contribution
    unique_files_list = list(dict.fromkeys(file_names))
    ep_max: Dict[str, Any] = {}
    for fname in unique_files_list:
        ep_mask = fn_arr == fname
        if ep_mask.sum() == 0:
            continue
        ep_max[fname] = {
            "count": int(ep_mask.sum()),
            "weight_sum": round(float(w[ep_mask].sum()), 4),
        }
    top_ep = sorted(ep_max.items(), key=lambda x: -x[1]["weight_sum"])[:3]

    return {
        "highest_weight_10pct_mass_fraction": round(top_w_mass, 4),
        "lowest_return_10pct_weight_share": round(low_ret_w_share, 4),
        "highest_return_10pct_weight_share": round(high_ret_w_share, 4),
        "largest_equal_return_group": {
            "return_value": round(float(largest_group_val), 6),
            "pointer_share": round(lg_pointer_share, 4),
            "weight_share": round(lg_weight_share, 4),
        },
        "gini_weight": round(gini, 4),
        "top_3_episodes": [
            {"file": k, "count": v["count"], "weight_sum": v["weight_sum"]}
            for k, v in top_ep
        ],
    }


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative array."""
    x = x[~np.isnan(x)]
    if len(x) == 0 or x.sum() <= 0:
        return 0.0
    x_sorted = np.sort(x)
    n = len(x)
    return float(
        (2 * np.arange(1, n + 1) * x_sorted).sum() / (n * x_sorted.sum()) - (n + 1) / n
    )


# ---------------------------------------------------------------------------
# Active-set simulation (profiler-specific)
# ---------------------------------------------------------------------------


def simulate_active_set(
    weights: np.ndarray,
    M: int,
    q_ret: np.ndarray,
    q_pos: np.ndarray,
    q_neg: np.ndarray,
    q_up: np.ndarray,
    q_down: np.ndarray,
    file_names: List[str],
    pointer_ids: Optional[Sequence[str]] = None,
    n_cycles: int = N_CYCLE_SEEDS,
) -> Dict[str, Any]:
    """Simulate weighted reservoir sampling over multiple cycles.

    Returns composition, replacement, inclusion-frequency and concentration
    metrics.
    """
    N = len(weights)
    if pointer_ids is None:
        pointer_ids = tuple(str(i) for i in range(N))
    effective_M = min(M, N)
    cycles: List[np.ndarray] = []
    inclusion_count = np.zeros(N, dtype=np.int64)
    for cycle_idx in range(n_cycles):
        indices = reservoir_sample(
            weights, effective_M, cycle_idx, pointer_ids=pointer_ids
        )
        cycles.append(indices)
        inclusion_count[indices] += 1

    inclusion_freq = inclusion_count / n_cycles
    concentration = float((inclusion_freq**2).sum())

    tag_labels = ["q_pos>0.5", "q_neg>0.5", "q_up>0", "q_down>0", "legacy_done"]
    tag_arrays = {
        "q_pos>0.5": (q_pos > 0.5).astype(np.float64),
        "q_neg>0.5": (q_neg > 0.5).astype(np.float64),
        "q_up>0": (q_up > 0).astype(np.float64),
        "q_down>0": (q_down > 0).astype(np.float64),
        "legacy_done": np.zeros(N, dtype=np.float64),
    }
    composition: Dict[str, Any] = {}
    for label, arr in tag_arrays.items():
        cycle_means = []
        for idxs in cycles:
            cycle_means.append(float(arr[idxs].mean()))
        comp = np.array(cycle_means)
        composition[label] = {
            "mean_pct": round(float(comp.mean() * 100), 2),
            "std_pct": round(float(comp.std() * 100), 4),
        }

    q_bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    ret_quantile_comp: Dict[str, Any] = {}
    for bin_index, (lo, hi) in enumerate(q_bins):
        label = f"qG_{lo:.0%}_{hi:.0%}"
        upper = q_ret <= hi if bin_index == len(q_bins) - 1 else q_ret < hi
        bin_mask = (q_ret >= lo) & upper & ~np.isnan(q_ret)
        if not bin_mask.any():
            continue
        cycle_fracs = []
        for idxs in cycles:
            n_in = int(bin_mask[idxs].sum())
            cycle_fracs.append(n_in / max(1, len(idxs)))
        frac = np.array(cycle_fracs)
        ret_quantile_comp[label] = {
            "mean_pct": round(float(frac.mean() * 100), 2),
            "std_pct": round(float(frac.std() * 100), 4),
        }
    missing_mask = np.isnan(q_ret)
    missing_fracs = [
        float(missing_mask[idxs].sum() / max(1, len(idxs))) for idxs in cycles
    ]
    ret_quantile_comp["unranked"] = {
        "mean_pct": round(float(np.mean(missing_fracs) * 100), 2),
        "std_pct": round(float(np.std(missing_fracs) * 100), 4),
    }

    unique_files = list(dict.fromkeys(file_names))
    ep_comp: Dict[str, Any] = {}
    for fname in unique_files:
        ep_mask = np.array([fn == fname for fn in file_names])
        cycle_fracs = []
        for idxs in cycles:
            n_ep = int(ep_mask[idxs].sum())
            cycle_fracs.append(float(n_ep))
        frac = np.array(cycle_fracs)
        ep_comp[fname] = {
            "mean_count": round(float(frac.mean()), 1),
            "std_count": round(float(frac.std()), 2),
        }

    replacement_fractions = []
    jaccard_distances = []
    for i in range(1, len(cycles)):
        intersection = np.intersect1d(cycles[i - 1], cycles[i], assume_unique=True)
        union = np.union1d(cycles[i - 1], cycles[i])
        jaccard = len(intersection) / max(1, len(union))
        replacement_fractions.append(1.0 - len(intersection) / effective_M)
        jaccard_distances.append(1.0 - jaccard)
    mean_replacement = (
        float(np.mean(replacement_fractions)) if replacement_fractions else 0.0
    )
    std_replacement = (
        float(np.std(replacement_fractions)) if replacement_fractions else 0.0
    )
    mean_jaccard_distance = (
        float(np.mean(jaccard_distances)) if jaccard_distances else 0.0
    )

    all_selected = np.unique(np.concatenate(cycles))
    unique_pct = round(len(all_selected) / N * 100, 2)

    return {
        "M": effective_M,
        "n_cycles": n_cycles,
        "tag_composition": composition,
        "ret_quantile_composition": ret_quantile_comp,
        "episode_composition": {
            sorted(ep_comp.items(), key=lambda x: -x[1]["mean_count"])[0][0]: (
                ep_comp[sorted(ep_comp.items(), key=lambda x: -x[1]["mean_count"])[0][0]]
            ),
            "top_3_episodes": sorted(ep_comp.items(), key=lambda x: -x[1]["mean_count"])[:3],
        },
        "mean_replacement_fraction": round(mean_replacement, 4),
        "std_replacement_fraction": round(std_replacement, 4),
        "mean_jaccard_distance": round(mean_jaccard_distance, 4),
        "unique_pointers_pct": unique_pct,
        "inclusion_concentration": round(float(concentration), 6),
    }


def is_approximately_uniform(
    weights: np.ndarray,
    M: int,
    n_cycles: int = 50,
    rtol: float = 0.15,
) -> bool:
    N = len(weights)
    effective_M = min(M, N)
    counts = np.zeros(N, dtype=np.int64)
    for cycle_idx in range(n_cycles):
        indices = reservoir_sample(weights, effective_M, cycle_idx)
        counts[indices] += 1
    freq = counts / n_cycles
    expected = effective_M / N
    if expected <= 0:
        return True
    cv = float(freq.std() / expected)
    return cv < rtol


# ---------------------------------------------------------------------------
# H / h sensitivity runner
# ---------------------------------------------------------------------------


def run_H_sensitivity(
    ret_arr: Dict[int, np.ndarray],
    d_up_pct: Dict[int, np.ndarray],
    d_down_pct: Dict[int, np.ndarray],
    legacy_done: np.ndarray,
    n_pointers: int,
    file_names: List[str],
    pointer_ids: Sequence[str],
    d_horizons: Sequence[int],
    fixed_h: int,
) -> Dict[str, Any]:
    """Run sensitivity grid for each return horizon.

    Returns per-H results with the declared selected_h fixed.
    """
    if fixed_h not in d_horizons:
        raise ValueError(f"fixed_h={fixed_h} is not in d_horizons")
    by_H: Dict[str, Any] = {}
    for H in sorted(ret_arr.keys()):
        qG = percentile_rank(ret_arr[H])
        q_pos = np.maximum(0.0, 2.0 * qG - 1.0)
        q_neg = np.maximum(0.0, 1.0 - 2.0 * qG)
        by_H[f"H={H}"] = run_sensitivity_grid(
            q_pos, q_neg,
            d_up_pct[fixed_h], d_down_pct[fixed_h],
            legacy_done, n_pointers,
            selected_H=H, selected_h=fixed_h,
            file_names=file_names, pointer_ids=pointer_ids, q_ret=qG,
        )
    return by_H


def run_h_sensitivity(
    ret_arr: Dict[int, np.ndarray],
    d_up_pct: Dict[int, np.ndarray],
    d_down_pct: Dict[int, np.ndarray],
    legacy_done: np.ndarray,
    n_pointers: int,
    file_names: List[str],
    pointer_ids: Sequence[str],
    horizons: Sequence[int],
    fixed_H: int,
) -> Dict[str, Any]:
    """Run sensitivity grid for each d_horizon.

    Returns per-h results with the declared selected_H fixed.
    """
    if fixed_H not in horizons:
        raise ValueError(f"fixed_H={fixed_H} is not in horizons")
    qG = percentile_rank(ret_arr[fixed_H])
    q_pos = np.maximum(0.0, 2.0 * qG - 1.0)
    q_neg = np.maximum(0.0, 1.0 - 2.0 * qG)
    by_h: Dict[str, Any] = {}
    for h in sorted(d_up_pct.keys()):
        by_h[f"h={h}"] = run_sensitivity_grid(
            q_pos, q_neg,
            d_up_pct[h], d_down_pct[h],
            legacy_done, n_pointers,
            selected_H=fixed_H, selected_h=h,
            file_names=file_names, pointer_ids=pointer_ids, q_ret=qG,
        )
    return by_h


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------

DEFAULT_SENSITIVITY_CONFIGS = SENSITIVITY_GRID


def profile_corpus(
    data_root: Union[str, Path],
    data_split_seed: int,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    d_horizons: Sequence[int] = DEFAULT_D_HORIZONS,
    selected_H: Optional[int] = None,
    selected_h: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the full corpus inventory for one split seed.

    Returns a JSON-serializable dict with:
    - corpus summary (files, transitions, eligible pointers)
    - per-horizon return distributions and quantiles
    - per-h smoothing directional change statistics
    - signal correlations
    - episode contributions by return quantile
    - corrected density/concentration metrics
    - effective sample size and active-set simulations
      for each candidate weight configuration
    - H/h sensitivity across all combinations
    """
    data_root = Path(data_root)
    train_files, val_files = collect_and_split(data_root, data_split_seed)

    train_set = set(train_files)
    val_set = set(val_files)
    assert train_set.isdisjoint(val_set), "Train/val files overlap"

    all_files = sorted(train_files)
    file_hashes_map: Dict[str, str] = {}
    total_episodes = len(all_files)
    total_transitions = 0

    # Phase 2: load every training file and extract per-pointer raw values
    file_names: List[str] = []
    file_hashes_list: List[str] = []
    pointer_ids: List[str] = []
    timesteps: List[int] = []
    immediate_rewards: List[float] = []
    legacy_done_flags: List[bool] = []
    returns: Dict[int, List[Optional[float]]] = {H: [] for H in horizons}
    d_vals: Dict[int, List[Optional[float]]] = {h: [] for h in d_horizons}
    episode_lengths: List[int] = []
    pointer_file_indices: List[int] = []

    for fi, fpath in enumerate(all_files):
        fhash = file_hash(fpath)
        file_hashes_map[str(fpath)] = fhash
        data = np.load(fpath)
        rewards = data["reward"].astype(np.float64)
        done_arr = data["done"]
        T = len(rewards)
        total_transitions += T
        episode_lengths.append(T)

        for H in horizons:
            ret = compute_factual_returns(rewards, done_arr, H)
            for t in range(T):
                val = ret[t] if not np.isnan(ret[t]) else None
                returns[H].append(val)

        for h in d_horizons:
            dv = compute_directional_change(rewards, done_arr, h)
            for t in range(T):
                val = dv[t] if not np.isnan(dv[t]) else None
                d_vals[h].append(val)

        for t in range(T):
            file_names.append(fpath.name)
            file_hashes_list.append(fhash)
            pointer_ids.append(f"{fhash}:{t}")
            timesteps.append(t)
            immediate_rewards.append(float(rewards[t]))
            legacy_done_flags.append(bool(done_arr[t]))
            pointer_file_indices.append(fi)

    n_pointers = len(file_names)

    # Phase 3: convert to numpy and compute percentile ranks
    ret_arr: Dict[int, np.ndarray] = {}
    ret_pct: Dict[int, np.ndarray] = {}
    for H in horizons:
        arr = np.array(returns[H], dtype=np.float64)
        ret_arr[H] = arr
        ret_pct[H] = percentile_rank(arr)

    d_arr: Dict[int, np.ndarray] = {}
    d_up_pct: Dict[int, np.ndarray] = {}
    d_down_pct: Dict[int, np.ndarray] = {}
    for h in d_horizons:
        arr = np.array(d_vals[h], dtype=np.float64)
        d_arr[h] = arr
        d_up_pct[h] = positive_tail_percentile(arr)
        d_down_pct[h] = negative_tail_percentile(arr)

    selected_H = max(horizons) if selected_H is None else selected_H
    selected_h = max(d_horizons) if selected_h is None else selected_h
    if selected_H not in horizons:
        raise ValueError(f"selected_H={selected_H} is not in horizons")
    if selected_h not in d_horizons:
        raise ValueError(f"selected_h={selected_h} is not in d_horizons")

    # Use one explicitly selected horizon for consistent q_pos/q_neg.
    qG_selected = ret_pct[selected_H]
    q_pos = np.maximum(0.0, 2.0 * qG_selected - 1.0)
    q_neg = np.maximum(0.0, 1.0 - 2.0 * qG_selected)

    q_up_selected = d_up_pct[selected_h]
    q_down_selected = d_down_pct[selected_h]

    legacy_done_arr = np.array(legacy_done_flags, dtype=np.float64)

    # Phase 4: quantile summaries
    def quantile_summary(arr: np.ndarray) -> Dict[str, float]:
        finite = arr[~np.isnan(arr)]
        if len(finite) == 0:
            return {"min": None, "q1": None, "median": None, "q3": None, "max": None, "mean": None, "std": None}
        return {
            "min": round(float(finite.min()), 6),
            "q1": round(float(np.percentile(finite, 25)), 6),
            "median": round(float(np.median(finite)), 6),
            "q3": round(float(np.percentile(finite, 75)), 6),
            "max": round(float(finite.max()), 6),
            "mean": round(float(finite.mean()), 6),
            "std": round(float(finite.std()), 6),
        }

    return_quantiles = {f"H={H}": quantile_summary(ret_arr[H]) for H in horizons}
    d_quantiles = {f"h={h}": quantile_summary(d_arr[h]) for h in d_horizons}
    tie_frequencies = {f"H={H}": compute_tie_frequencies(ret_arr[H]) for H in horizons}

    # Phase 5: correlation analysis
    correlations = compute_signal_correlations(
        ret_arr, d_arr, q_pos, q_neg, d_up_pct, d_down_pct,
    )

    # Phase 6: episode contribution by return quantile
    episode_contrib = compute_episode_contributions(
        file_names, qG_selected,
    )

    # Phase 7: sensitivity grid with active-set simulation
    sensitivity = run_sensitivity_grid(
        q_pos, q_neg,
        q_up_selected, q_down_selected,
        legacy_done_arr, n_pointers,
        selected_H=selected_H, selected_h=selected_h,
        file_names=file_names, pointer_ids=pointer_ids, q_ret=qG_selected,
    )

    # Phase 8: H/h sensitivity across all combinations
    H_sensitivity = run_H_sensitivity(
        ret_arr, d_up_pct, d_down_pct,
        legacy_done_arr, n_pointers, file_names, pointer_ids, d_horizons,
        fixed_h=selected_h,
    )
    h_sensitivity = run_h_sensitivity(
        ret_arr, d_up_pct, d_down_pct,
        legacy_done_arr, n_pointers, file_names, pointer_ids, horizons,
        fixed_H=selected_H,
    )

    # Phase 9: equal-return crowding sensitivity on the balanced config.
    balanced_w_uncorrected = compute_weights(
        q_pos, q_neg,
        q_up_selected, q_down_selected,
        legacy_done_arr,
        0.1, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
    )
    crowding_sensitivity: Dict[str, Any] = {}
    for rho in CROWDING_RHO_CANDIDATES:
        corrected_weights = apply_equal_return_crowding(
            balanced_w_uncorrected,
            ret_arr[selected_H],
            eta=0.1,
            rho=rho,
        )
        active_sets = {
            str(M): simulate_active_set(
                corrected_weights,
                M,
                qG_selected,
                q_pos,
                q_neg,
                q_up_selected,
                q_down_selected,
                file_names,
                pointer_ids,
            )
            for M in ACTIVE_SET_M_CANDIDATES
            if M <= n_pointers
        }
        crowding_sensitivity[str(rho)] = {
            "rho": rho,
            "effective_sample_size": round(
                effective_sample_size(corrected_weights), 1
            ),
            "ess_ratio": round(
                effective_sample_size(corrected_weights) / n_pointers, 4
            ),
            "density_metrics": compute_density_metrics(
                corrected_weights, ret_arr[selected_H], file_names
            ),
            "active_set_simulation": active_sets,
        }

    selected_weights = apply_equal_return_crowding(
        balanced_w_uncorrected,
        ret_arr[selected_H],
        eta=0.1,
        rho=DEFAULT_CROWDING_RHO,
    )
    density_metrics = compute_density_metrics(
        selected_weights, ret_arr[selected_H], file_names
    )

    # Phase 10: per-horizon eligibility
    eligible_counts = {}
    for H in horizons:
        eligible_counts[f"H={H}"] = int((~np.isnan(ret_arr[H])).sum())

    # Phase 11: surprise counts per d_horizon
    surprise_counts = {}
    for h in d_horizons:
        n_up = int((d_up_pct[h] > 0).sum())
        n_down = int((d_down_pct[h] > 0).sum())
        pct_up = round(n_up / max(1, n_pointers) * 100, 2)
        pct_down = round(n_down / max(1, n_pointers) * 100, 2)
        surprise_counts[f"h={h}"] = {
            "up_count": n_up, "up_pct": pct_up,
            "down_count": n_down, "down_pct": pct_down,
        }

    any_legacy_done = any(legacy_done_flags)
    corpus_summary: Dict[str, Any] = {
        "seed": data_split_seed,
        "data_root": str(data_root),
        "n_files": len(all_files),
        "n_val_files": len(val_files),
        "n_episodes": total_episodes,
        "n_transitions": total_transitions,
        "n_eligible_pointers": n_pointers,
        "file_hashes": file_hashes_map,
        "train_val_disjoint": train_set.isdisjoint(val_set),
        "episode_lengths": episode_lengths,
        "episode_length_summary": quantile_summary(np.array(episode_lengths, dtype=np.float64)),
        "immediate_reward_summary": quantile_summary(np.array(immediate_rewards, dtype=np.float64)),
        "eligible_counts": eligible_counts,
        "return_quantiles": return_quantiles,
        "tie_frequencies": tie_frequencies,
        "d_quantiles": d_quantiles,
        "surprise_counts": surprise_counts,
        "quantize_decimals": QUANTIZE_DECIMALS,
        "signal_correlations": correlations,
        "episode_contributions": episode_contrib,
        "density_metrics": density_metrics,
        "crowding_sensitivity": crowding_sensitivity,
        "selected_crowding_rho": DEFAULT_CROWDING_RHO,
        "sensitivity_grid": sensitivity,
        "sensitivity_by_H": H_sensitivity,
        "sensitivity_by_h": h_sensitivity,
        "selected_H": selected_H,
        "selected_h": selected_h,
        "any_legacy_done": any_legacy_done,
        "note_episode_end": (
            "The rollout schema stores done = terminated OR truncated "
            "without distinction.  Separate terminated/truncated information "
            "is unavailable for these files.  The field is named 'legacy_done' "
            "to avoid claiming the terminated/truncated distinction.  "
            f"legacy_done=True samples: {int(legacy_done_arr.sum())}."
        ),
    }

    return corpus_summary


def run_seed_profile_and_save(
    data_root: Union[str, Path],
    data_split_seed: int,
    out_dir: Union[str, Path],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    d_horizons: Sequence[int] = DEFAULT_D_HORIZONS,
) -> Path:
    """Profile one seed and write JSON results to ``out_dir / seed_N / corpus_summary.json``."""
    out_dir = Path(out_dir)
    seed_dir = out_dir / f"seed{data_split_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    summary = profile_corpus(data_root, data_split_seed, horizons, d_horizons)

    out_path = seed_dir / "corpus_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_fallback)
    return out_path


def _json_fallback(obj: Any) -> str:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def load_summary(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a saved corpus summary JSON."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataclass for structured report metadata
# ---------------------------------------------------------------------------


@dataclass
class ProfilerMetadata:
    data_root: Path
    data_split_seeds: List[int]
    horizons: Tuple[int, ...]
    d_horizons: Tuple[int, ...]
    sensitivity_configs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
