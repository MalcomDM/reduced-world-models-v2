"""Shared priority formulas — extracted for use by both
corpus_profiler.py and the Stage-7.0B factual index.

Kept separate to avoid circular imports and enable exact numerical parity
regression tests between the profiler and the archive.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Constants  (canonical — keep in sync with corpus_profiler)
# ---------------------------------------------------------------------------

QUANTIZE_DECIMALS: int = 8
EPSILON: float = 1e-8


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def quantize(x: np.ndarray, decimals: int = QUANTIZE_DECIMALS) -> np.ndarray:
    return np.round(x.astype(np.float64), decimals)


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Per-pointer metric computations
# ---------------------------------------------------------------------------


def compute_factual_returns(
    rewards: np.ndarray,
    done: np.ndarray,
    horizon: int,
) -> np.ndarray:
    T = len(rewards)
    out = np.full(T, np.nan, dtype=np.float64)
    if T == 0 or horizon < 1:
        return out
    max_t = T - horizon
    for t in range(max_t + 1):
        check_done = done[t : t + horizon - 1] if horizon > 1 else np.array([], dtype=bool)
        if not check_done.any():
            out[t] = quantize(rewards[t : t + horizon].sum())
    return out


def compute_directional_change(
    rewards: np.ndarray,
    done: np.ndarray,
    h: int,
) -> np.ndarray:
    T = len(rewards)
    out = np.full(T, np.nan, dtype=np.float64)
    if T < 2 * h:
        return out
    for t in range(h, T - h + 1):
        check_done = done[t - h : t + h - 1] if h > 0 else np.array([], dtype=bool)
        if not check_done.any():
            pre = rewards[t - h : t]
            post = rewards[t : t + h]
            out[t] = float(post.mean() - pre.mean())
    return out


# ---------------------------------------------------------------------------
# Percentile ranks
# ---------------------------------------------------------------------------


def percentile_rank(values: np.ndarray) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    finite_mask = ~np.isnan(values)
    if not finite_mask.any():
        return out
    valid = values[finite_mask].copy()
    n = len(valid)
    if n == 0:
        return out
    q_valid = quantize(valid)
    order = np.argsort(q_valid)
    rank = np.empty(n, dtype=np.float64)
    rank[order] = np.arange(n, dtype=np.float64)
    tie_start = 0
    while tie_start < n:
        tie_end = tie_start
        while tie_end < n and q_valid[order[tie_end]] == q_valid[order[tie_start]]:
            tie_end += 1
        mean_rank = float(np.mean(rank[order[tie_start:tie_end]]))
        rank[order[tie_start:tie_end]] = mean_rank
        tie_start = tie_end
    pct = rank / (n - 1) if n > 1 else 0.5
    out[finite_mask] = pct
    return out


def positive_tail_percentile(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    pos_mask = values > 0
    if not pos_mask.any():
        return out
    out[pos_mask] = percentile_rank(values[pos_mask])
    return out


def negative_tail_percentile(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    neg_mask = values < 0
    if not neg_mask.any():
        return out
    out[neg_mask] = percentile_rank(-values[neg_mask])
    return out


# ---------------------------------------------------------------------------
# Priority scoring — canonical config
# ---------------------------------------------------------------------------

DEFAULT_SELECTED_H: int = 12
DEFAULT_SELECTED_h: int = 4  # selected h (directional change window)
DEFAULT_ETA: float = 0.1
DEFAULT_LAMBDA_POS: float = 1.0
DEFAULT_LAMBDA_NEG: float = 1.0
DEFAULT_LAMBDA_UP: float = 1.0
DEFAULT_LAMBDA_DOWN: float = 1.0
DEFAULT_LAMBDA_LEGACY_DONE: float = 0.0
DEFAULT_ALPHA: float = 1.0
DEFAULT_BETA: float = 1.0
DEFAULT_CROWDING_RHO: float = 0.25
DEFAULT_ACTIVE_SET_M: int = 1024


def compute_priority_score(
    q_pos: np.ndarray,
    q_neg: np.ndarray,
    q_up: np.ndarray,
    q_down: np.ndarray,
    legacy_done: np.ndarray,
    lambda_pos: float = DEFAULT_LAMBDA_POS,
    lambda_neg: float = DEFAULT_LAMBDA_NEG,
    lambda_up: float = DEFAULT_LAMBDA_UP,
    lambda_down: float = DEFAULT_LAMBDA_DOWN,
    lambda_legacy_done: float = DEFAULT_LAMBDA_LEGACY_DONE,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    eps: float = EPSILON,
) -> np.ndarray:
    """Raw priority score (before uniform mixture or crowding).

    ``score_i = lambda_pos * (eps + q_pos)^alpha
              + lambda_neg * (eps + q_neg)^alpha
              + lambda_up * (eps + q_up)^beta
              + lambda_down * (eps + q_down)^beta
              + lambda_legacy_done * legacy_done_i``
    """
    score = np.zeros_like(q_pos, dtype=np.float64)
    score += lambda_pos * np.where(np.isnan(q_pos), 0.0, (eps + q_pos) ** alpha)
    score += lambda_neg * np.where(np.isnan(q_neg), 0.0, (eps + q_neg) ** alpha)
    score += lambda_up * np.where(np.isnan(q_up), 0.0, (eps + q_up) ** beta)
    score += lambda_down * np.where(np.isnan(q_down), 0.0, (eps + q_down) ** beta)
    score += lambda_legacy_done * np.where(np.isnan(legacy_done), 0.0, legacy_done)
    return score


def apply_equal_return_crowding(
    score: np.ndarray,
    factual_returns: np.ndarray,
    rho: float = DEFAULT_CROWDING_RHO,
    eps: float = EPSILON,
) -> np.ndarray:
    """Divide each pointer's score by ``count_same_quantized_return ** rho``.

    Pointers with NaN return retain their original score.  The output is
    a non-negative finite array that can be fed straight into
    ``compute_probabilities``.
    """
    if rho < 0:
        raise ValueError(f"rho must be non-negative, got {rho}")
    out = score.astype(np.float64, copy=True)
    finite_ret = np.isfinite(factual_returns)
    if rho == 0 or not finite_ret.any():
        return out
    grouped = quantize(factual_returns[finite_ret])
    _, inverse, counts = np.unique(grouped, return_inverse=True, return_counts=True)
    group_counts = counts[inverse].astype(np.float64)
    out[finite_ret] /= np.maximum(np.power(group_counts, rho), eps)
    return out


def compute_probabilities(
    crowded_score: np.ndarray,
    eta: float = DEFAULT_ETA,
    eps: float = EPSILON,
) -> np.ndarray:
    """Convert crowded scores to a probability simplex with uniform floor.

    ``P(i) = eta / N + (1 - eta) * crowded_score_i / sum(crowded_score)``

    If all crowded scores are zero (or eta == 1), returns exact uniform
    ``1/N`` for all pointers.

    Every returned probability is finite, strictly positive and sums to
    1 within machine tolerance.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must be in [0, 1], got {eta}")
    if np.any(~np.isfinite(crowded_score)) or np.any(crowded_score < 0):
        raise ValueError("crowded_score must contain finite non-negative values")
    n = len(crowded_score)
    if n == 0:
        return np.array([], dtype=np.float64)
    if eta == 1.0:
        return np.full(n, 1.0 / n, dtype=np.float64)
    total = crowded_score.sum()
    if total <= 0:
        return np.full(n, 1.0 / n, dtype=np.float64)
    priority = crowded_score / total
    return eta / n + (1.0 - eta) * priority


def compute_weights(
    q_pos: np.ndarray,
    q_neg: np.ndarray,
    q_up: np.ndarray,
    q_down: np.ndarray,
    legacy_done: np.ndarray,
    factual_returns: np.ndarray,
    eta: float = DEFAULT_ETA,
    lambda_pos: float = DEFAULT_LAMBDA_POS,
    lambda_neg: float = DEFAULT_LAMBDA_NEG,
    lambda_up: float = DEFAULT_LAMBDA_UP,
    lambda_down: float = DEFAULT_LAMBDA_DOWN,
    lambda_legacy_done: float = DEFAULT_LAMBDA_LEGACY_DONE,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    rho: float = DEFAULT_CROWDING_RHO,
) -> np.ndarray:
    """End-to-end: score → crowding → probabilities.

    This is the canonical entry point used by the index builder.  The
    profiler's version (which additionally handles sensitivity over
    many configs) uses the same underlying primitives.
    """
    score = compute_priority_score(
        q_pos, q_neg, q_up, q_down, legacy_done,
        lambda_pos, lambda_neg, lambda_up, lambda_down,
        lambda_legacy_done, alpha, beta,
    )
    crowded = apply_equal_return_crowding(score, factual_returns, rho)
    return compute_probabilities(crowded, eta)


# ---------------------------------------------------------------------------
# Sampling primitives
# ---------------------------------------------------------------------------


def _gumbel_key(weight: float, u: float) -> float:
    if weight <= 0 or np.isnan(weight):
        return np.inf
    return -np.log(u) / weight


@lru_cache(maxsize=256)
def _stable_uniforms(
    cycle_seed: int,
    pointer_ids: Tuple[str, ...],
) -> np.ndarray:
    uniforms = np.empty(len(pointer_ids), dtype=np.float64)
    for i, pointer_id in enumerate(pointer_ids):
        digest = hashlib.sha256(
            f"{cycle_seed}\0{pointer_id}".encode("utf-8")
        ).digest()
        integer = int.from_bytes(digest[:8], "big") >> 11
        uniforms[i] = (integer + 0.5) / float(1 << 53)
    uniforms.setflags(write=False)
    return uniforms


def reservoir_sample(
    weights: np.ndarray,
    M: int,
    cycle_seed: int,
    pointer_ids: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Weighted reservoir sampling without replacement.

    ``key_i = -log(U_i) / weight_i`` where ``U_i`` is derived from
    SHA-256 of ``(cycle_seed, pointer_id)``.  Returns sorted indices.
    """
    N = len(weights)
    if M < 1:
        raise ValueError(f"M must be positive, got {M}")
    if pointer_ids is None:
        pointer_ids = tuple(str(i) for i in range(N))
    pointer_ids = tuple(pointer_ids)
    if len(pointer_ids) != N:
        raise ValueError("pointer_ids length must match weights")
    if len(set(pointer_ids)) != N:
        raise ValueError("pointer_ids must be unique")
    valid_count = int(np.sum(np.isfinite(weights) & (weights > 0)))
    if M > valid_count:
        raise ValueError(
            f"M={M} exceeds the {valid_count} pointers with positive finite weight"
        )
    if M >= N:
        return np.arange(N)
    M = int(M)
    keys = np.full(N, np.inf, dtype=np.float64)
    valid = np.isfinite(weights) & (weights > 0)
    uniforms = _stable_uniforms(cycle_seed, pointer_ids)
    keys[valid] = -np.log(uniforms[valid]) / weights[valid]
    order = np.argsort(keys)
    selected = order[:M]
    selected.sort()
    return selected


def effective_sample_size(weights: np.ndarray) -> float:
    r"""ESS = (\sum w_i)^2 / \sum w_i^2."""
    w = weights[~np.isnan(weights)]
    if len(w) == 0 or w.sum() <= 0:
        return 0.0
    return float(w.sum() ** 2 / (w * w).sum())
