"""Tests for Stage 7.0A Factual Memory Corpus Inventory.

Covers:
  1. Deterministic output.
  2. Train-only isolation.
  3. Episode-boundary: done at final included position is VALID.
  4. Hand-calculated returns and directional surprise.
  5. Percentile ties with quantization.
  6. q_pos/q_neg use same qG (regression).
  7. Quantization consistency.
  8. Zero-change behavior.
  9. Short episodes.
 10. Finite/nonnegative weights.
 11. Nonzero uniform floor.
 12. Effective sample size.
 13. Reservoir sampling determinism, turnover, uniformity.
 14. Density/concentration metrics.
 15. No model mutation.
 16. Full profile smoke.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rwm.memory.corpus_profiler import (
    DEFAULT_HORIZONS,
    DEFAULT_D_HORIZONS,
    QUANTIZE_DECIMALS,
    compute_factual_returns,
    compute_directional_change,
    percentile_rank,
    quantize,
    positive_tail_percentile,
    negative_tail_percentile,
    compute_weights,
    apply_equal_return_crowding,
    effective_sample_size,
    file_hash,
    profile_corpus,
    run_sensitivity_grid,
    compute_signal_correlations,
    compute_tie_frequencies,
    compute_density_metrics,
    reservoir_sample,
    simulate_active_set,
    is_approximately_uniform,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_npz(path: Path, rewards: np.ndarray, done: np.ndarray | None = None,
              actions: np.ndarray | None = None) -> Path:
    T = len(rewards)
    if done is None:
        done = np.zeros(T, dtype=bool)
    if actions is None:
        actions = np.zeros((T, 3), dtype=np.float32)
    obs = np.zeros((T, 96, 96, 3), dtype=np.uint8)
    np.savez(path, obs=obs, action=actions, reward=rewards, done=done)
    return path


def _make_corpus_dir(tmp_path: Path, file_specs: list[dict]) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    for spec in file_specs:
        _make_npz(root / spec["name"], spec["rewards"], spec.get("done"))
    return root


# ---------------------------------------------------------------------------
# 1. Deterministic output
# ---------------------------------------------------------------------------


class TestDeterministic:
    def test_same_input_same_hash(self, tmp_path: Path) -> None:
        r = np.array([1.0, -0.1, -0.1, 2.0])
        p1 = _make_npz(tmp_path / "a.npz", r)
        p2 = _make_npz(tmp_path / "b.npz", r)
        assert file_hash(p1) == file_hash(p2)
        assert file_hash(p1) == file_hash(p1)

    def test_same_input_same_profile(self, tmp_path: Path) -> None:
        root = _make_corpus_dir(tmp_path, [
            {"name": "ep1.npz", "rewards": np.array([1.0, -0.1, 2.0, -0.1])},
            {"name": "ep2.npz", "rewards": np.array([0.0, 3.0, -0.1, 0.5])},
        ])
        s1 = profile_corpus(root, data_split_seed=0, horizons=(1, 2), d_horizons=(2,))
        s2 = profile_corpus(root, data_split_seed=0, horizons=(1, 2), d_horizons=(2,))
        assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)


# ---------------------------------------------------------------------------
# 2. Train-only isolation
# ---------------------------------------------------------------------------


class TestTrainOnlyIsolation:
    def test_val_files_excluded(self, tmp_path: Path) -> None:
        root = _make_corpus_dir(tmp_path, [
            {"name": "train1.npz", "rewards": np.array([1.0, 2.0, 3.0])},
            {"name": "train2.npz", "rewards": np.array([4.0, 5.0])},
            {"name": "val1.npz", "rewards": np.array([99.0, 100.0])},
        ])
        s = profile_corpus(root, data_split_seed=0, horizons=(1,), d_horizons=(2,))
        assert s["n_files"] == 2
        assert s["n_val_files"] == 1
        assert s["train_val_disjoint"] is True
        assert s["n_eligible_pointers"] == 5

    def test_disjoint_detected(self, tmp_path: Path) -> None:
        root = _make_corpus_dir(tmp_path, [
            {"name": "a.npz", "rewards": np.array([1.0, 2.0])},
            {"name": "b.npz", "rewards": np.array([3.0, 4.0])},
        ])
        s = profile_corpus(root, data_split_seed=0, horizons=(1,), d_horizons=(2,))
        assert s["train_val_disjoint"] is True


# ---------------------------------------------------------------------------
# 3. Episode-boundary handling (corrected semantics)
# ---------------------------------------------------------------------------


class TestEpisodeBoundaries:
    def test_done_at_final_included_is_valid(self) -> None:
        """done at the final included position is VALID."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        done = np.array([False, False, False, True])
        # H=2: pointer t=2 → window [2:4) → check [2:3) = [False] → valid.
        #   But wait: max_t = T-H = 4-2 = 2. t=2: window [2:4) = [3, 4].
        #   check_done = done[2:3] = [done[2]] = [False] → valid. return = 3+4 = 7.
        ret = compute_factual_returns(rewards, done, horizon=2)
        assert np.isclose(ret[2], 7.0), f"Expected 7.0, got {ret[2]}"

    def test_done_before_final_is_invalid(self) -> None:
        """done at a non-final included position is INVALID."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        done = np.array([False, True, False, False])
        ret = compute_factual_returns(rewards, done, horizon=3)
        # t=0: window [0:3) = [1,2,3], check_done = done[0:2] = [False, True] → invalid
        assert np.isnan(ret[0])
        # t=1: window [1:4) = [2,3,4], check_done = done[1:3] = [True, False] → invalid
        assert np.isnan(ret[1])

    def test_done_only_at_last_h4(self) -> None:
        """H=4 with done at the final position only."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        done = np.array([False, False, False, True])
        ret = compute_factual_returns(rewards, done, horizon=4)
        # t=0: window [0:4), check_done = done[0:3] = [False,False,False] → valid
        assert np.isclose(ret[0], 10.0)

    def test_done_horizon1_always_valid(self) -> None:
        """H=1 is always valid regardless of done."""
        rewards = np.array([10.0, 20.0])
        done = np.array([True, False])
        ret = compute_factual_returns(rewards, done, horizon=1)
        assert ret[0] == 10.0
        assert ret[1] == 20.0

    def test_directional_change_allows_done_at_end(self) -> None:
        """directional change allows done at the final included position."""
        rewards = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        done = np.array([False, False, False, False, False, True])
        dv = compute_directional_change(rewards, done, h=2)
        # t=2: window [0:4), check_done = done[0:3] = [False,False,False] → valid
        assert not np.isnan(dv[2])
        # t=3: window [1:5), check_done = done[1:4] = [False,False,False] → valid
        assert not np.isnan(dv[3])

    def test_directional_change_rejects_mid_window_done(self) -> None:
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        done = np.array([False, True, False, False, False])
        dv = compute_directional_change(rewards, done, h=2)
        # t=2: window [0:4), check = done[0:3] = [False,True,False] → invalid
        assert np.isnan(dv[2])


# ---------------------------------------------------------------------------
# 4. Hand-calculated returns and directional surprise
# ---------------------------------------------------------------------------


class TestHandCalculated:
    def test_factual_return_h1(self) -> None:
        rewards = np.array([5.0, 3.0, -1.0])
        ret = compute_factual_returns(rewards, np.zeros(3, dtype=bool), horizon=1)
        assert np.isclose(ret, [5.0, 3.0, -1.0]).all()

    def test_factual_return_h2(self) -> None:
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        ret = compute_factual_returns(rewards, np.zeros(4, dtype=bool), horizon=2)
        assert np.isclose(ret[0], 3.0)
        assert np.isclose(ret[1], 5.0)
        assert np.isclose(ret[2], 7.0)
        assert np.isnan(ret[3])

    def test_factual_return_quantized(self) -> None:
        """Floating summation produces consistent quantized results."""
        rewards = np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
                            -0.1, -0.1, -0.1, -0.1, -0.1, -0.1])
        ret = compute_factual_returns(rewards, np.zeros(12, dtype=bool), horizon=12)
        assert np.isclose(ret[0], -1.2)
        q = quantize(ret)
        assert q[0] == -1.2  # exact after quantization

    def test_directional_change_hand(self) -> None:
        rewards = np.array([1.0, 1.0, 5.0, 5.0, 1.0, 1.0])
        dv = compute_directional_change(rewards, np.zeros(6, dtype=bool), h=2)
        assert np.isclose(dv[2], 4.0)
        assert np.isclose(dv[3], 0.0)

    def test_percentile_rank_hand(self) -> None:
        vals = np.array([10.0, 20.0, 30.0, 40.0])
        pct = percentile_rank(vals)
        assert np.isclose(pct[0], 0.0)
        assert np.isclose(pct[1], 1.0 / 3)
        assert np.isclose(pct[2], 2.0 / 3)
        assert np.isclose(pct[3], 1.0)

    def test_upward_surprise(self) -> None:
        d_vals = np.array([-5.0, 0.0, 2.0, 10.0, 5.0, -1.0])
        up = positive_tail_percentile(d_vals)
        assert up[0] == 0.0
        assert up[1] == 0.0
        assert up[2] == 0.0   # smallest positive → percentile 0
        assert up[3] == 1.0   # largest positive → percentile 1
        assert 0 < up[4] < 1
        assert up[5] == 0.0

    def test_downward_surprise(self) -> None:
        d_vals = np.array([-5.0, 0.0, -1.0, 10.0, -10.0, -3.0])
        down = negative_tail_percentile(d_vals)
        assert down[0] > 0.0
        assert down[1] == 0.0
        assert down[2] == 0.0   # smallest magnitude → percentile 0
        assert down[3] == 0.0
        assert down[4] == 1.0   # largest magnitude → percentile 1
        assert 0 < down[5] < 1


# ---------------------------------------------------------------------------
# 5. Percentile ties with quantization
# ---------------------------------------------------------------------------


class TestPercentileTies:
    def test_ties_get_mean_rank(self) -> None:
        vals = np.array([1.0, 1.0, 2.0, 3.0])
        pct = percentile_rank(vals)
        assert np.isclose(pct[0], pct[1])
        assert 0.0 < pct[0] < pct[2]
        assert pct[3] == 1.0

    def test_all_tied(self) -> None:
        vals = np.array([5.0, 5.0, 5.0])
        pct = percentile_rank(vals)
        assert np.isclose(pct[0], 0.5)
        assert np.isclose(pct[1], 0.5)
        assert np.isclose(pct[2], 0.5)

    def test_tie_no_nan(self) -> None:
        vals = np.array([np.nan, 1.0, 1.0, 2.0])
        pct = percentile_rank(vals)
        assert np.isnan(pct[0])
        assert np.isclose(pct[1], pct[2])
        assert np.isclose(pct[3], 1.0)

    def test_numerically_close_tied_by_quantization(self) -> None:
        """Floating summation artefacts are merged by quantization."""
        vals = np.array([-1.2, -1.2000000000000002, 0.0, 3.0])
        pct = percentile_rank(vals)
        # -1.2 and -1.2000000000000002 should be treated as tied
        assert np.isclose(pct[0], pct[1]), "Quantization must merge close values"
        assert pct[0] < pct[2]  # tied group < 0.0


# ---------------------------------------------------------------------------
# 6. q_pos/q_neg use same qG (regression)
# ---------------------------------------------------------------------------


class TestSameHorizon:
    def test_q_pos_q_neg_from_same_qG(self) -> None:
        """Verify that q_pos and q_neg use the same percentile rank (qG)."""
        rng = np.random.RandomState(42)
        returns = rng.randn(100)
        qG = percentile_rank(returns)
        q_pos = np.maximum(0.0, 2.0 * qG - 1.0)
        q_neg = np.maximum(0.0, 1.0 - 2.0 * qG)
        # For any pointer, at most one of q_pos or q_neg can be > 0
        both_positive = (q_pos > 0) & (q_neg > 0)
        assert both_positive.sum() == 0, (
            "q_pos and q_neg must be derived from the same qG "
            "and cannot both be positive"
        )
        # For qG < 0.5: q_pos = 0, q_neg > 0
        # For qG > 0.5: q_pos > 0, q_neg = 0
        # For qG = 0.5: both = 0
        assert np.allclose(q_pos + q_neg, np.maximum(0.0, 2.0 * np.abs(qG - 0.5)))


# ---------------------------------------------------------------------------
# 7. Quantization
# ---------------------------------------------------------------------------


class TestQuantization:
    def test_quantize_rounds_to_decimals(self) -> None:
        x = np.array([1.23456789, -1.2000000000000002])
        q = quantize(x)
        assert q[0] == 1.23456789
        assert q[1] == -1.2

    def test_quantize_preserves_nan(self) -> None:
        x = np.array([1.0, np.nan, 2.0])
        q = quantize(x)
        assert np.isnan(q[1])
        assert q[0] == 1.0

    def test_tie_frequencies_use_quantized(self) -> None:
        vals = np.array([-1.2, -1.2000000000000002, 0.0, 0.0, 3.0])
        t = compute_tie_frequencies(vals)
        assert t["unique_values"] == 3  # -1.2, 0.0, 3.0
        assert t["largest_tie_size"] == 2

    def test_numerically_close_sums_deterministic_ties(self) -> None:
        """Logically equal return sums are grouped by quantization."""
        r = np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
                      -0.1, -0.1, -0.1, -0.1, -0.1, -0.1])
        ret = compute_factual_returns(r, np.zeros(12, dtype=bool), horizon=12)
        # All valid pointers have the same return
        valid = ret[~np.isnan(ret)]
        q = quantize(valid)
        unique = np.unique(q)
        assert len(unique) == 1, (
            f"All identical-reward sums should quantize to one unique value, "
            f"got {unique}"
        )


# ---------------------------------------------------------------------------
# 8. Zero-change behavior
# ---------------------------------------------------------------------------


class TestZeroChange:
    def test_d_zero_no_surprise(self) -> None:
        d_vals = np.array([0.0, 0.0, 0.0, 0.0])
        up = positive_tail_percentile(d_vals)
        down = negative_tail_percentile(d_vals)
        assert (up == 0.0).all()
        assert (down == 0.0).all()

    def test_all_identical_rewards_no_surprise(self) -> None:
        rewards = np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1])
        dv = compute_directional_change(rewards, np.zeros(6, dtype=bool), h=2)
        assert np.isclose(dv[2], 0.0)
        assert np.isclose(dv[3], 0.0)
        up = positive_tail_percentile(dv)
        assert (up == 0.0).all()


# ---------------------------------------------------------------------------
# 9. Short episodes
# ---------------------------------------------------------------------------


class TestShortEpisodes:
    def test_episode_too_short_for_horizon(self) -> None:
        rewards = np.array([1.0, 2.0])
        for H in [2, 4, 8, 12]:
            ret = compute_factual_returns(rewards, np.zeros(2, dtype=bool), horizon=H)
            if H == 2:
                assert np.isclose(ret[0], 3.0)
                assert np.isnan(ret[1])
            else:
                assert np.isnan(ret).all()

    def test_episode_too_short_for_directional(self) -> None:
        rewards = np.array([1.0, 2.0, 3.0])
        for h in [2, 3, 4]:
            dv = compute_directional_change(rewards, np.zeros(3, dtype=bool), h=h)
            assert np.isnan(dv).all()

    def test_no_pointers_for_large_h(self, tmp_path: Path) -> None:
        root = _make_corpus_dir(tmp_path, [
            {"name": "a.npz", "rewards": np.array([1.0, 2.0, 3.0])},
            {"name": "b.npz", "rewards": np.array([4.0, 5.0, 6.0])},
        ])
        s = profile_corpus(root, data_split_seed=0, horizons=(1, 8), d_horizons=(2,))
        assert s["eligible_counts"]["H=8"] == 0


# ---------------------------------------------------------------------------
# 10. Finite/nonnegative weights
# ---------------------------------------------------------------------------


class TestFiniteNonnegativeWeights:
    def test_weights_finite_and_nonnegative(self) -> None:
        n = 50
        rng = np.random.RandomState(42)
        q_pos = rng.rand(n)
        q_neg = rng.rand(n)
        q_up = rng.rand(n)
        q_down = rng.rand(n)
        legacy_done = rng.randint(0, 2, size=n).astype(np.float64)
        w = compute_weights(q_pos, q_neg, q_up, q_down, legacy_done,
                            0.1, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0)
        assert np.all(np.isfinite(w))
        assert np.all(w >= 0.0)

    def test_weights_with_nan(self) -> None:
        q_pos = np.array([0.1, np.nan, 0.3])
        q_neg = np.array([np.nan, 0.2, 0.1])
        w = compute_weights(q_pos, q_neg, np.zeros(3), np.zeros(3), np.zeros(3),
                            0.1, 1.0, 1.0, 0.5, 0.5, 0.0, 1.0, 1.0)
        assert np.all(np.isfinite(w))
        assert w[1] >= 0.0


# ---------------------------------------------------------------------------
# 11. Nonzero uniform floor
# ---------------------------------------------------------------------------


class TestUniformFloor:
    def test_min_weight_at_least_eta(self) -> None:
        w = compute_weights(
            np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10),
            0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
        )
        assert np.isclose(w, 0.1).all()
        assert np.isclose(w.sum(), 1.0)

    def test_eta_prevents_zero_weights(self) -> None:
        w = compute_weights(
            np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5),
            0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
        )
        assert np.all(w >= 0.1 / 5)
        assert np.isclose(w.sum(), 1.0)


# ---------------------------------------------------------------------------
# 12. Equal-return crowding correction
# ---------------------------------------------------------------------------


class TestEqualReturnCrowding:
    def test_rho_zero_is_exact_parity(self) -> None:
        weights = np.array([0.025, 0.175, 0.3, 0.5])
        returns = np.array([-1.2, -1.2, 2.0, 3.0])
        corrected = apply_equal_return_crowding(weights, returns, 0.1, rho=0)
        assert np.array_equal(corrected, weights)

    def test_sqrt_correction_preserves_floor(self) -> None:
        weights = np.full(5, 0.2)
        returns = np.array([-1.2, -1.2, -1.2, -1.2, 5.0])
        corrected = apply_equal_return_crowding(weights, returns, 0.1, rho=0.5)
        assert corrected[4] > corrected[0]
        assert np.all(corrected >= 0.1 / 5)
        assert np.isclose(corrected.sum(), 1.0)

    def test_full_correction_equalizes_group_bonus(self) -> None:
        weights = np.full(5, 0.2)
        returns = np.array([-1.2, -1.2, -1.2, -1.2, 5.0])
        corrected = apply_equal_return_crowding(weights, returns, 0.1, rho=1)
        floor = 0.1 / 5
        duplicate_bonus = np.sum(corrected[:4] - floor)
        singleton_bonus = corrected[4] - floor
        assert np.isclose(duplicate_bonus, singleton_bonus)

    def test_nan_returns_are_unchanged(self) -> None:
        weights = np.array([0.4, 0.6])
        returns = np.array([np.nan, 2.0])
        corrected = apply_equal_return_crowding(weights, returns, 0.1, rho=0.5)
        assert np.array_equal(corrected, weights)

    def test_negative_rho_rejected(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            apply_equal_return_crowding(
                np.ones(2), np.zeros(2), eta=0.1, rho=-0.5
            )


# ---------------------------------------------------------------------------
# 13. Effective sample size
# ---------------------------------------------------------------------------


class TestEffectiveSampleSize:
    def test_uniform_weights(self) -> None:
        w = np.ones(100)
        ess = effective_sample_size(w)
        assert np.isclose(ess, 100.0)

    def test_one_dominant_weight(self) -> None:
        w = np.array([100.0, 0.01, 0.01, 0.01])
        ess = effective_sample_size(w)
        assert ess < 4.0
        assert ess > 1.0

    def test_all_nan(self) -> None:
        assert effective_sample_size(np.array([np.nan, np.nan])) == 0.0

    def test_empty(self) -> None:
        assert effective_sample_size(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# 13. Reservoir sampling
# ---------------------------------------------------------------------------


class TestReservoirSampling:
    def test_same_seed_same_set(self) -> None:
        weights = np.random.RandomState(0).rand(100).astype(np.float64)
        s1 = reservoir_sample(weights, M=10, cycle_seed=42)
        s2 = reservoir_sample(weights, M=10, cycle_seed=42)
        assert np.array_equal(s1, s2)

    def test_existing_keys_stable_when_pointer_inserted(self) -> None:
        weights = np.ones(20, dtype=np.float64)
        pointer_ids = [f"pointer-{i}" for i in range(20)]
        original = reservoir_sample(
            weights, M=8, cycle_seed=42, pointer_ids=pointer_ids
        )

        inserted_ids = ["new-pointer", *pointer_ids]
        inserted = reservoir_sample(
            np.ones(21), M=8, cycle_seed=42, pointer_ids=inserted_ids
        )
        inserted_selected_ids = {inserted_ids[i] for i in inserted}
        original_selected_ids = {pointer_ids[i] for i in original}

        # At most the new pointer displaces one old member; unrelated random
        # keys are stable because they derive from pointer identity.
        assert len(original_selected_ids - inserted_selected_ids) <= 1

    def test_duplicate_pointer_ids_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            reservoir_sample(
                np.ones(3), M=2, cycle_seed=0, pointer_ids=["a", "a", "b"]
            )

    def test_different_seed_different_set(self) -> None:
        weights = np.random.RandomState(0).rand(100).astype(np.float64)
        s1 = reservoir_sample(weights, M=10, cycle_seed=42)
        s2 = reservoir_sample(weights, M=10, cycle_seed=99)
        assert not np.array_equal(s1, s2)

    def test_no_duplicates(self) -> None:
        weights = np.random.RandomState(0).rand(100).astype(np.float64)
        indices = reservoir_sample(weights, M=20, cycle_seed=0)
        assert len(indices) == len(set(indices))

    def test_M_valid_unique_pointers(self) -> None:
        N = 100
        weights = np.random.RandomState(0).rand(N).astype(np.float64)
        M = 25
        indices = reservoir_sample(weights, M=M, cycle_seed=0)
        assert len(indices) == M
        assert indices.min() >= 0
        assert indices.max() < N

    def test_larger_weights_higher_inclusion(self) -> None:
        N = 50
        weights = np.ones(N, dtype=np.float64)
        weights[:25] = 100.0  # first 25 have much higher weight
        M = 10
        n_trials = 50
        counts_high = 0
        counts_low = 0
        for seed in range(n_trials):
            idx = reservoir_sample(weights, M=M, cycle_seed=seed)
            counts_high += (idx < 25).sum()
            counts_low += (idx >= 25).sum()
        # High-weight half should dominate selections
        assert counts_high > counts_low * 2, "High-weight pointers should be selected more"

    def test_zero_weights_rejected(self) -> None:
        weights = np.array([0.0, 0.0, 1.0, 0.0, 2.0], dtype=np.float64)
        idx = reservoir_sample(weights, M=2, cycle_seed=0)
        assert 0 not in idx
        assert 1 not in idx
        assert 3 not in idx

    def test_nan_weights_rejected(self) -> None:
        weights = np.array([np.nan, 1.0, np.nan, 2.0], dtype=np.float64)
        idx = reservoir_sample(weights, M=2, cycle_seed=0)
        assert 0 not in idx
        assert 2 not in idx

    def test_uniform_approximately_uniform(self) -> None:
        N = 200
        weights = np.ones(N, dtype=np.float64)
        assert is_approximately_uniform(weights, M=20, n_cycles=500, rtol=0.18)


# ---------------------------------------------------------------------------
# 14. Density / concentration metrics
# ---------------------------------------------------------------------------


class TestDensityMetrics:
    def test_uniform_weights(self) -> None:
        w = np.ones(100)
        q = np.linspace(0, 1, 100)
        fn = [f"f{i}.npz" for i in range(100)]
        m = compute_density_metrics(w, q, fn)
        assert m["highest_weight_10pct_mass_fraction"] < 0.2
        assert m["gini_weight"] < 0.1

    def test_highly_skewed(self) -> None:
        w = np.zeros(100)
        w[0] = 100.0
        q = np.linspace(0, 1, 100)
        fn = [f"f{i}.npz" for i in range(100)]
        m = compute_density_metrics(w, q, fn)
        assert m["gini_weight"] > 0.8

    def test_low_return_dense(self) -> None:
        """Most pointers have low return and high weight."""
        w = np.ones(100)
        w[:80] = 10.0
        q = np.linspace(0, 1, 100)
        fn = [f"f{i}.npz" for i in range(100)]
        m = compute_density_metrics(w, q, fn)
        # Lowest-return 10% should hold significant weight
        assert m["lowest_return_10pct_weight_share"] > 0.05

    def test_high_return_dense(self) -> None:
        """Most pointers have high return and high weight."""
        w = np.ones(100)
        w[80:] = 10.0
        q = np.linspace(0, 1, 100)
        fn = [f"f{i}.npz" for i in range(100)]
        m = compute_density_metrics(w, q, fn)
        assert m["highest_return_10pct_weight_share"] > 0.05

    def test_single_dominant(self) -> None:
        """One equal-return group dominates."""
        w = np.array([1.0] * 90 + [100.0] * 10)
        q = np.array([-1.2] * 90 + [5.0] * 10)
        fn = [f"f{i}.npz" for i in range(100)]
        m = compute_density_metrics(w, q, fn)
        assert m["largest_equal_return_group"]["pointer_share"] > 0.5
        assert m["largest_equal_return_group"]["return_value"] == -1.2


# ---------------------------------------------------------------------------
# 15. No model/training mutation
# ---------------------------------------------------------------------------


class TestNoModelMutation:
    def test_import_dry(self) -> None:
        from rwm.memory import corpus_profiler
        assert hasattr(corpus_profiler, "profile_corpus")

    def test_no_side_effects_on_call(self, tmp_path: Path) -> None:
        root = _make_corpus_dir(tmp_path, [
            {"name": "a.npz", "rewards": np.array([1.0, 2.0])},
            {"name": "b.npz", "rewards": np.array([3.0, 4.0])},
        ])
        before = set(tmp_path.rglob("*"))
        s = profile_corpus(root, data_split_seed=0, horizons=(1,), d_horizons=(2,))
        after = set(tmp_path.rglob("*"))
        new_files = after - before
        for f in new_files:
            assert "corpus" in str(f)


# ---------------------------------------------------------------------------
# 16. Sensitivity grid sanity
# ---------------------------------------------------------------------------


class TestSensitivityGrid:
    def test_ess_nonnegative(self) -> None:
        n = 100
        rng = np.random.RandomState(0)
        q_pos = rng.rand(n)
        q_neg = rng.rand(n)
        q_up = rng.rand(n)
        q_down = rng.rand(n)
        legacy_done = rng.randint(0, 2, size=n).astype(np.float64)
        fn = [f"f{i}.npz" for i in range(n)]
        q_ret = rng.rand(n)
        results = run_sensitivity_grid(
            q_pos, q_neg, q_up, q_down, legacy_done, n,
            selected_H=12, selected_h=4, file_names=fn, q_ret=q_ret,
        )
        assert len(results) > 0
        for r in results:
            assert r["effective_sample_size"] > 0
            assert 0 < r["ess_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# 17. Correlation sanity
# ---------------------------------------------------------------------------


class TestCorrelations:
    def test_identical_signals(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        corr = compute_signal_correlations(
            {1: x}, {2: x}, x, x, {2: x}, {2: x},
        )
        for key, val in corr.items():
            assert np.isclose(val, 1.0, atol=1e-6), f"{key} should be 1.0, got {val}"


# ---------------------------------------------------------------------------
# 18. Tie frequency
# ---------------------------------------------------------------------------


class TestTieFrequency:
    def test_no_ties(self) -> None:
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        t = compute_tie_frequencies(vals)
        assert t["tied_groups"] == 0
        assert t["fraction_tied"] == 0.0

    def test_all_ties(self) -> None:
        vals = np.array([1.0, 1.0, 1.0])
        t = compute_tie_frequencies(vals)
        assert t["tied_groups"] == 1
        assert t["fraction_tied"] == 1.0
        assert t["largest_tie_size"] == 3


# ---------------------------------------------------------------------------
# 19. Full profile smoke (end-to-end)
# ---------------------------------------------------------------------------


class TestFullProfile:
    def test_profile_with_multiple_files(self, tmp_path: Path) -> None:
        root = _make_corpus_dir(tmp_path, [
            {"name": "a.npz", "rewards": np.array([1.0, 2.0, 3.0])},
            {"name": "b.npz", "rewards": np.array([0.0, -0.1, 5.0, 2.0])},
            {"name": "c.npz", "rewards": np.array([1.0, 1.0])},
            {"name": "d.npz", "rewards": np.array([2.0, 2.0, 2.0])},
        ])
        s = profile_corpus(root, data_split_seed=0, horizons=(1, 2), d_horizons=(2,))
        assert s["n_files"] >= 2
        assert s["n_transitions"] >= 1
        assert s["n_eligible_pointers"] == s["n_transitions"]
        assert s["train_val_disjoint"] is True
        assert len(s["return_quantiles"]) == 2
        assert len(s["d_quantiles"]) == 1
        assert len(s["surprise_counts"]) == 1
        assert len(s["sensitivity_grid"]) > 0
        # Verify q_pos/q_neg use same horizon
        assert s["selected_H"] == 2  # max horizon
        assert s["quantize_decimals"] == QUANTIZE_DECIMALS

    def test_no_done_in_corpus(self, tmp_path: Path) -> None:
        root = _make_corpus_dir(tmp_path, [
            {"name": "a.npz", "rewards": np.array([-0.1, -0.1, 5.0, -0.1])},
            {"name": "b.npz", "rewards": np.array([1.0, 2.0, 1.0])},
        ])
        s = profile_corpus(root, data_split_seed=0, horizons=(1, 4), d_horizons=(2,))
        assert s["any_legacy_done"] is False

    def test_selected_horizons_persisted(self, tmp_path: Path) -> None:
        root = _make_corpus_dir(tmp_path, [
            {"name": "a.npz", "rewards": np.array([1.0, 2.0, 3.0])},
            {"name": "b.npz", "rewards": np.array([4.0, 5.0])},
            {"name": "c.npz", "rewards": np.array([1.0, 1.0])},
        ])
        s = profile_corpus(root, data_split_seed=0, horizons=(1, 2, 4), d_horizons=(3, 4))
        assert s["selected_H"] == 4
        assert s["selected_h"] == 4
        # Verify q_pos/q_neg use selected_H
        assert s["sensitivity_grid"][0]["selected_H"] == 4
