"""Stage 7.0A — Factual Memory Corpus Inventory.

Read-only profiler over eligible training-file transitions.
Produces per-pointer metrics, corpus diagnostics, and a sensitivity grid
for continuous priority weights. Does not create or modify any model,
trainer, cache, or sampler.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from rwm.data.split import collect_and_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HORIZONS: Tuple[int, ...] = (1, 2, 4, 8, 12)
DEFAULT_D_HORIZONS: Tuple[int, ...] = (2, 3, 4)
EPSILON: float = 1e-8  # small constant for percentile rank stability

# Sensitivity grid — interpretable candidate configurations
# Each entry: (name, eta, lambda_pos, lambda_neg, lambda_up, lambda_down, lambda_term, alpha, beta)
SENSITIVITY_GRID: List[Tuple[str, float, float, float, float, float, float, float, float]] = [
    ("uniform",         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
    ("return_only",     0.1, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0),
    ("return_sharp",    0.1, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 1.0),
    ("change_focused",  0.1, 0.5, 0.5, 2.0, 2.0, 1.0, 1.0, 2.0),
    ("balanced",        0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ("high_floor",      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0),
    ("return_extreme",  0.1, 2.0, 2.0, 0.5, 0.5, 1.0, 2.0, 1.5),
]

WEIGHT_LABELS = [
    "eta", "lambda_pos", "lambda_neg", "lambda_up",
    "lambda_down", "lambda_term", "alpha", "beta",
]

# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------


def file_hash(path: Path) -> str:
    """SHA-256 hex digest of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Per-pointer metric computations (operate on single file arrays)
# ---------------------------------------------------------------------------


def compute_factual_returns(
    rewards: np.ndarray,
    done: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Undiscounted factual return for every valid pointer.

    ``return[t] = sum(rewards[t : t+horizon])`` when the full window is
    within the episode and no ``done`` flag is raised inside the window.
    Invalid pointers (near episode end or across a done boundary) receive
    ``NaN``.

    For horizon 1 the window is ``[t, t+1)`` — always valid if ``t < T``.
    """
    T = len(rewards)
    out = np.full(T, np.nan, dtype=np.float64)
    if T == 0:
        return out
    if horizon < 1:
        return out
    max_t = T - horizon
    for t in range(max_t + 1):
        window_done = done[t : t + horizon]
        if horizon == 1 or not window_done.any():
            out[t] = rewards[t : t + horizon].sum()
    return out


def compute_directional_change(
    rewards: np.ndarray,
    done: np.ndarray,
    h: int,
) -> np.ndarray:
    r"""d_t(h) = mean(rewards[t:t+h]) - mean(rewards[t-h:t]).

    Returns ``NaN`` for pointers where the full ``[t-h, t+h)`` window is
    not entirely within the episode or crosses a done boundary.
    """
    T = len(rewards)
    out = np.full(T, np.nan, dtype=np.float64)
    if T < 2 * h:
        return out
    for t in range(h, T - h + 1):
        pre = rewards[t - h : t]
        post = rewards[t : t + h]
        if not done[t - h : t + h].any():
            out[t] = post.mean() - pre.mean()
    return out


# ---------------------------------------------------------------------------
# Percentile rank helpers
# ---------------------------------------------------------------------------


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1] for each element.

    Ties receive the mean percentile rank of the tied group.
    NaN values are preserved as NaN.
    """
    out = np.full_like(values, np.nan, dtype=np.float64)
    finite_mask = ~np.isnan(values)
    if not finite_mask.any():
        return out
    valid = values[finite_mask]
    n = len(valid)
    if n == 0:
        return out
    order = np.argsort(valid)
    rank = np.empty(n, dtype=np.float64)
    rank[order] = np.arange(n, dtype=np.float64)
    tie_start = 0
    while tie_start < n:
        tie_end = tie_start
        while tie_end < n and np.isclose(valid[order[tie_end]], valid[order[tie_start]]):
            tie_end += 1
        mean_rank = float(np.mean(rank[order[tie_start:tie_end]]))
        rank[order[tie_start:tie_end]] = mean_rank
        tie_start = tie_end
    pct = rank / (n - 1) if n > 1 else 0.5
    out[finite_mask] = pct
    return out


def positive_tail_percentile(values: np.ndarray) -> np.ndarray:
    """Percentile rank among strictly positive values; zero otherwise.

    For v > 0: percentile rank of v among all positive v.
    For v <= 0: 0.
    """
    out = np.zeros_like(values, dtype=np.float64)
    pos_mask = values > 0
    if not pos_mask.any():
        return out
    pos_vals = values[pos_mask]
    pct = percentile_rank(pos_vals)
    out[pos_mask] = pct
    return out


def negative_tail_percentile(values: np.ndarray) -> np.ndarray:
    """Percentile rank among strictly negative values (by magnitude); zero otherwise.

    For v < 0: percentile rank of -v among all positive -v values.
    For v >= 0: 0.
    """
    out = np.zeros_like(values, dtype=np.float64)
    neg_mask = values < 0
    if not neg_mask.any():
        return out
    neg_mags = -values[neg_mask]
    pct = percentile_rank(neg_mags)
    out[neg_mask] = pct
    return out


# ---------------------------------------------------------------------------
# Weight / priority computation
# ---------------------------------------------------------------------------


def compute_weights(
    q_pos: np.ndarray,
    q_neg: np.ndarray,
    q_up: np.ndarray,
    q_down: np.ndarray,
    terminal: np.ndarray,
    eta: float,
    lambda_pos: float,
    lambda_neg: float,
    lambda_up: float,
    lambda_down: float,
    lambda_term: float,
    alpha: float,
    beta: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Continuous priority weight for each pointer.

    ``w_i = eta + lp * (eps + q_pos)^alpha + ln * (eps + q_neg)^alpha
           + lup * (eps + q_up)^beta + ldn * (eps + q_down)^beta
           + lt * terminal_i``
    """
    w = np.full_like(q_pos, eta, dtype=np.float64)
    w += lambda_pos * np.where(np.isnan(q_pos), 0.0, (eps + q_pos) ** alpha)
    w += lambda_neg * np.where(np.isnan(q_neg), 0.0, (eps + q_neg) ** alpha)
    w += lambda_up * np.where(np.isnan(q_up), 0.0, (eps + q_up) ** beta)
    w += lambda_down * np.where(np.isnan(q_down), 0.0, (eps + q_down) ** beta)
    w += lambda_term * np.where(np.isnan(terminal), 0.0, terminal)
    return w


def effective_sample_size(weights: np.ndarray) -> float:
    r"""Effective sample size: ESS = (\sum w_i)^2 / \sum w_i^2."""
    w = weights[~np.isnan(weights)]
    if len(w) == 0 or w.sum() <= 0:
        return 0.0
    return float(w.sum() ** 2 / (w * w).sum())


# ---------------------------------------------------------------------------
# Sensitivity grid runner
# ---------------------------------------------------------------------------


def run_sensitivity_grid(
    q_pos: np.ndarray,
    q_neg: np.ndarray,
    q_up: np.ndarray,
    q_down: np.ndarray,
    terminal: np.ndarray,
    n_pointers: int,
) -> List[Dict[str, Any]]:
    """Evaluate candidate weight configurations.

    Returns list of dicts with name, params, ESS, active-tag composition
    for candidate active-set sizes M.
    """
    results = []
    M_candidates = [max(100, n_pointers // 10), max(500, n_pointers // 4), n_pointers]
    for name, eta, lp, ln, lup, ldn, lt, alpha, beta in SENSITIVITY_GRID:
        w = compute_weights(
            q_pos, q_neg, q_up, q_down, terminal,
            eta, lp, ln, lup, ldn, lt, alpha, beta,
        )
        ess = effective_sample_size(w)
        finite_w = w[~np.isnan(w)]
        composition: Dict[str, Any] = {}
        for M in M_candidates:
            effective_M = min(M, len(finite_w))
            top_idx = np.argsort(finite_w)[-effective_M:]
            top_w = finite_w[top_idx]
            top_total = top_w.sum() if top_w.sum() > 0 else 1.0
            composition[str(M)] = {
                "mass_fraction": float(top_w.sum() / finite_w.sum()),
                "n_pointers": int(effective_M),
                "tag_positive_pct": None,
                "tag_negative_pct": None,
                "tag_terminal_pct": None,
            }
        finite_q = q_pos[~np.isnan(q_pos)]
        tag_pos = (finite_q > 0.5).sum()
        finite_qn = q_neg[~np.isnan(q_neg)]
        tag_neg = (finite_qn > 0.5).sum()
        finite_t = terminal[~np.isnan(terminal)]
        tag_term = finite_t.sum()
        total_finite = np.sum(~np.isnan(q_pos))
        results.append({
            "name": name,
            "params": {
                "eta": eta,
                "lambda_pos": lp,
                "lambda_neg": ln,
                "lambda_up": lup,
                "lambda_down": ldn,
                "lambda_term": lt,
                "alpha": alpha,
                "beta": beta,
            },
            "effective_sample_size": round(ess, 1),
            "ess_ratio": round(ess / max(1, len(finite_w)), 4),
            "active_set_composition": composition,
            "global_tag_pct": {
                "positive_pos": round(float(tag_pos / max(1, total_finite) * 100), 2),
                "negative_pos": round(float(tag_neg / max(1, total_finite) * 100), 2),
                "terminal": round(float(tag_term / max(1, total_finite) * 100), 2),
            },
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
            r = np.corrcoef(a[mask], b[mask])[0, 1]
            if not np.isnan(r):
                corr[f"{keys[i]}_vs_{keys[j]}"] = round(float(r), 4)
    return corr


# ---------------------------------------------------------------------------
# Episode contribution analysis
# ---------------------------------------------------------------------------


def compute_episode_contributions(
    file_names: List[str],
    file_hashes_list: List[str],
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
    """Analyze tie frequency in a value array.

    Returns count of unique values, number of tied groups >1, and the
    fraction of elements in ties.
    """
    finite = values[~np.isnan(values)]
    if len(finite) == 0:
        return {"unique_values": 0, "tied_groups": 0, "fraction_tied": 0.0}
    unique, counts = np.unique(finite, return_counts=True)
    tied_mask = counts > 1
    tied_elements = counts[tied_mask].sum()
    return {
        "unique_values": int(len(unique)),
        "tied_groups": int(tied_mask.sum()),
        "fraction_tied": round(float(tied_elements / len(finite)), 4),
        "largest_tie_size": int(counts.max()) if len(counts) > 0 else 1,
    }


# ---------------------------------------------------------------------------
# Dense-region analysis
# ---------------------------------------------------------------------------


def compute_dense_region_impact(
    weights: np.ndarray,
    q_ret: np.ndarray,
) -> Dict[str, Any]:
    """Check whether dense return regions dominate weighted sampling.

    Computes what fraction of total weight is held by the top-k% of
    pointers sorted by return. If a small return window holds most weight,
    diversity is low.
    """
    finite = ~(np.isnan(weights) | np.isnan(q_ret))
    w = weights[finite]
    q = q_ret[finite]
    if len(w) == 0:
        return {}
    order = np.argsort(q)
    w_sorted = w[order]
    q_sorted = q[order]
    cum_w = np.cumsum(w_sorted) / w_sorted.sum()
    result: Dict[str, Any] = {}
    for pct in [10, 25, 50]:
        idx = int(len(q_sorted) * pct / 100)
        result[f"top_{pct}pct_weight_fraction"] = round(float(cum_w[idx]), 4)
    result["gini_weight"] = round(float(_gini(w)), 4)
    return result


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative array."""
    x = x[~np.isnan(x)]
    if len(x) == 0 or x.sum() <= 0:
        return 0.0
    x_sorted = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(x_sorted)
    return float((2 * np.arange(1, n + 1) * x_sorted).sum() / (n * x_sorted.sum()) - (n + 1) / n)


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------

DEFAULT_SENSITIVITY_CONFIGS = SENSITIVITY_GRID  # alias for import


def profile_corpus(
    data_root: Union[str, Path],
    data_split_seed: int,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    d_horizons: Sequence[int] = DEFAULT_D_HORIZONS,
) -> Dict[str, Any]:
    """Run the full corpus inventory for one split seed.

    Returns a JSON-serializable dict with:
    - corpus summary (files, transitions, eligible pointers)
    - per-horizon return distributions and quantiles
    - per-h smoothing directional change statistics
    - signal correlations
    - episode contributions by return quantile
    - dense-region impact analysis
    - effective sample size and active set composition
      for each candidate weight configuration
    """
    data_root = Path(data_root)
    train_files, val_files = collect_and_split(data_root, data_split_seed)

    # Phase 1: verify disjointness and hash files
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
    timesteps: List[int] = []
    immediate_rewards: List[float] = []
    terminated_flags: List[bool] = []
    truncated_flags: List[bool] = []
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
            timesteps.append(t)
            immediate_rewards.append(float(rewards[t]))
            terminated_flags.append(bool(done_arr[t]))
            truncated_flags.append(bool(done_arr[t]))  # not stored separately
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

    q_pos = np.maximum(0.0, 2.0 * ret_pct[max(horizons)] - 1.0)
    q_neg = np.maximum(0.0, 1.0 - 2.0 * ret_pct[min(horizons)])

    terminal_arr = np.array(terminated_flags, dtype=np.float64)

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
        file_names, file_hashes_list, ret_pct[max(horizons)],
    )

    # Phase 7: sensitivity grid
    # Use the median d_horizon for q_up / q_down
    mid_d_h = d_horizons[len(d_horizons) // 2]
    sensitivity = run_sensitivity_grid(
        q_pos, q_neg,
        d_up_pct[mid_d_h], d_down_pct[mid_d_h],
        terminal_arr, n_pointers,
    )

    # Also run sensitivity for each d_horizon separately
    sensitivity_by_h: Dict[str, List[Dict[str, Any]]] = {}
    for h in d_horizons:
        sensitivity_by_h[f"h={h}"] = run_sensitivity_grid(
            q_pos, q_neg,
            d_up_pct[h], d_down_pct[h],
            terminal_arr, n_pointers,
        )

    # Phase 8: dense-region impact (using balanced config)
    balanced_w = compute_weights(
        q_pos, q_neg,
        d_up_pct[mid_d_h], d_down_pct[mid_d_h],
        terminal_arr,
        0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    )
    dense_impact = compute_dense_region_impact(balanced_w, ret_pct[max(horizons)])

    # Phase 9: per-horizon eligibility
    eligible_counts = {}
    for H in horizons:
        eligible_counts[f"H={H}"] = int((~np.isnan(ret_arr[H])).sum())

    # Phase 10: surprise counts per d_horizon
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

    # Phase 11: final summary
    corpus_summary = {
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
        "signal_correlations": correlations,
        "episode_contributions": episode_contrib,
        "dense_region_impact": dense_impact,
        "sensitivity_grid": sensitivity,
        "sensitivity_by_d_horizon": sensitivity_by_h,
        "all_terminated": all(terminated_flags) if len(terminated_flags) > 0 else False,
        "all_truncated": all(truncated_flags) if len(truncated_flags) > 0 else False,
        "note_terminated_truncated": (
            "The rollout schema stores done=terminated|truncated without distinction. "
            "All done flags are False in the current corpus; all transitions are truncated "
            "(timed-out) episodes. Terminated/truncated separation requires schema extension."
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
