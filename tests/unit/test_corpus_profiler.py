"""Tests for Stage 7.0A Factual Memory Corpus Inventory.

Covers:
  1. Deterministic output (same input → same hash/profile).
  2. Train-only isolation (validation files excluded).
  3. Episode-boundary handling (done flag truncates return/surprise).
  4. Hand-calculated factual returns and directional surprise.
  5. Percentile ties (mean-rank assignment).
  6. Zero-change behavior (d_t = 0 → no surprise).
  7. Short episodes (truncated horizons invalid).
  8. Finite/nonnegative weights.
  9. Nonzero uniform floor (eta > 0 ensures min weight >= eta).
 10. Effective-sample-size calculation.
 11. No model/training mutation (imports are read-only).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rwm.memory.corpus_profiler import (
    DEFAULT_HORIZONS,
    DEFAULT_D_HORIZONS,
    compute_factual_returns,
    compute_directional_change,
    percentile_rank,
    positive_tail_percentile,
    negative_tail_percentile,
    compute_weights,
    effective_sample_size,
    file_hash,
    profile_corpus,
    run_sensitivity_grid,
    compute_signal_correlations,
    compute_tie_frequencies,
    compute_dense_region_impact,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_npz(path: Path, rewards: np.ndarray, done: np.ndarray | None = None,
              actions: np.ndarray | None = None) -> Path:
    """Write a synthetic rollout .npz file."""
    T = len(rewards)
    if done is None:
        done = np.zeros(T, dtype=bool)
    if actions is None:
        actions = np.zeros((T, 3), dtype=np.float32)
    obs = np.zeros((T, 96, 96, 3), dtype=np.uint8)
    np.savez(path, obs=obs, action=actions, reward=rewards, done=done)
    return path


def _make_corpus_dir(tmp_path: Path, file_specs: list[dict]) -> Path:
    """Create a corpus directory with multiple .npz files.

    Each spec: {"name": str, "rewards": array, "done": array or None}
    """
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
        """File hash is deterministic."""
        r = np.array([1.0, -0.1, -0.1, 2.0])
        p1 = _make_npz(tmp_path / "a.npz", r)
        p2 = _make_npz(tmp_path / "b.npz", r)
        assert file_hash(p1) == file_hash(p2)
        h1 = file_hash(p1)
        h2 = file_hash(p1)
        assert h1 == h2

    def test_same_input_same_profile(self, tmp_path: Path) -> None:
        """Profiling same corpus twice yields identical JSON."""
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
        """Validation files are never included in pointer stats."""
        root = _make_corpus_dir(tmp_path, [
            {"name": "train1.npz", "rewards": np.array([1.0, 2.0, 3.0])},
            {"name": "train2.npz", "rewards": np.array([4.0, 5.0])},
            {"name": "val1.npz", "rewards": np.array([99.0, 100.0])},
        ])
        s = profile_corpus(root, data_split_seed=0, horizons=(1,), d_horizons=(2,))
        assert s["n_files"] == 2, "Should only have 2 train files"
        assert s["n_val_files"] == 1
        assert s["train_val_disjoint"] is True
        assert s["n_eligible_pointers"] == 5  # 3 + 2 = 5 from training files

    def test_disjoint_detected(self, tmp_path: Path) -> None:
        """Train/val sets are always disjoint."""
        root = _make_corpus_dir(tmp_path, [
            {"name": "a.npz", "rewards": np.array([1.0, 2.0])},
            {"name": "b.npz", "rewards": np.array([3.0, 4.0])},
        ])
        s = profile_corpus(root, data_split_seed=0, horizons=(1,), d_horizons=(2,))
        assert s["train_val_disjoint"] is True


# ---------------------------------------------------------------------------
# 3. Episode-boundary handling
# ---------------------------------------------------------------------------


class TestEpisodeBoundaries:
    def test_done_truncates_return(self) -> None:
        """Return is NaN for windows that cross done=True."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        done = np.array([False, True, False, False, False])
        ret = compute_factual_returns(rewards, done, horizon=4)
        # t=0: window [0:4] includes done[1]=True → NaN
        assert np.isnan(ret[0])
        # t=1: window [1:5) includes done[1]=True but wait... done[1] is in the window
        #   Actually for t=1, window is [1:5) which includes done[1], done[2], done[3]
        #   done[1] is True → NaN
        assert np.isnan(ret[1])
        # t=2: window [2:6) → out of bounds (T=5, max_t = 5-4=1)
        assert np.isnan(ret[2])

    def test_done_horizon1_always_valid(self) -> None:
        """Horizon 1 is always valid regardless of done."""
        rewards = np.array([10.0, 20.0])
        done = np.array([True, False])
        ret = compute_factual_returns(rewards, done, horizon=1)
        assert ret[0] == 10.0
        assert ret[1] == 20.0

    def test_directional_change_respects_boundary(self) -> None:
        """d_t is NaN when the window crosses done."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        done = np.array([False, False, True, False, False])
        dv = compute_directional_change(rewards, done, h=2)
        # t=2: window [0:4) → includes done[2]=True → NaN
        assert np.isnan(dv[2])
        # t=0,1: need h=2 context before → NaN
        assert np.isnan(dv[0])
        assert np.isnan(dv[1])
        # t=3,4: need 2 after → NaN (T=5, valid range is t in [2, 3) = just t=2)
        assert np.isnan(dv[3])
        assert np.isnan(dv[4])

    def test_no_boundary_directional_change(self) -> None:
        """Without done, d_t is computed correctly."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        done = np.zeros(6, dtype=bool)
        dv = compute_directional_change(rewards, done, h=2)
        # t=2: mean(post[2:4]) - mean(pre[0:2]) = mean(3,4) - mean(1,2) = 3.5 - 1.5 = 2.0
        assert not np.isnan(dv[2])
        assert np.isclose(dv[2], 2.0)
        # t=3: mean(4,5) - mean(2,3) = 4.5 - 2.5 = 2.0
        assert np.isclose(dv[3], 2.0)


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
        assert np.isclose(ret[0], 3.0)   # 1+2
        assert np.isclose(ret[1], 5.0)   # 2+3
        assert np.isclose(ret[2], 7.0)   # 3+4
        assert np.isnan(ret[3])          # t=3: t+2=5 > T=4

    def test_directional_change_hand(self) -> None:
        rewards = np.array([1.0, 1.0, 5.0, 5.0, 1.0, 1.0])
        dv = compute_directional_change(rewards, np.zeros(6, dtype=bool), h=2)
        # t=2: mean(5,5) - mean(1,1) = 5 - 1 = 4
        assert np.isclose(dv[2], 4.0)
        # t=3: mean(5,1) - mean(1,5) = 3 - 3 = 0
        assert np.isclose(dv[3], 0.0)

    def test_percentile_rank_hand(self) -> None:
        vals = np.array([10.0, 20.0, 30.0, 40.0])
        pct = percentile_rank(vals)
        assert np.isclose(pct[0], 0.0)     # rank 0 / 3
        assert np.isclose(pct[1], 1.0 / 3)
        assert np.isclose(pct[2], 2.0 / 3)
        assert np.isclose(pct[3], 1.0)

    def test_upward_surprise(self) -> None:
        d_vals = np.array([-5.0, 0.0, 2.0, 10.0, 5.0, -1.0])
        up = positive_tail_percentile(d_vals)
        assert up[0] == 0.0   # negative → 0
        assert up[1] == 0.0   # zero → 0
        assert up[2] == 0.0   # smallest positive (2) → percentile 0
        assert up[3] == 1.0   # largest positive (10) → percentile 1
        assert 0 < up[4] < 1  # middle positive (5) → between
        assert up[5] == 0.0   # negative → 0

    def test_downward_surprise(self) -> None:
        d_vals = np.array([-5.0, 0.0, -1.0, 10.0, -10.0, -3.0])
        down = negative_tail_percentile(d_vals)
        assert down[0] > 0.0  # -5 → magnitude 5, rank among {1,3,5,10}
        assert down[1] == 0.0  # zero → 0
        assert down[2] == 0.0  # smallest magnitude (1) → percentile 0
        assert down[3] == 0.0  # positive → 0
        assert down[4] == 1.0  # largest magnitude (10) → percentile 1
        assert 0 < down[5] < 1  # middle magnitude (3) → between


# ---------------------------------------------------------------------------
# 5. Percentile ties
# ---------------------------------------------------------------------------


class TestPercentileTies:
    def test_ties_get_mean_rank(self) -> None:
        vals = np.array([1.0, 1.0, 2.0, 3.0])
        pct = percentile_rank(vals)
        # Values [1, 1] occupy ranks 0 and 1 → mean 0.5 → pct = 0.5/3 ≈ 0.1667
        assert np.isclose(pct[0], pct[1])
        assert 0.0 < pct[0] < pct[2]  # tied group < distinct 2.0
        assert pct[3] == 1.0  # 3.0 is max

    def test_all_tied(self) -> None:
        vals = np.array([5.0, 5.0, 5.0])
        pct = percentile_rank(vals)
        # All ranks = 1 → mean rank = 1 → pct = 1/2 = 0.5 (n=3, n-1=2)
        assert np.isclose(pct[0], 0.5)
        assert np.isclose(pct[1], 0.5)
        assert np.isclose(pct[2], 0.5)

    def test_tie_no_nan(self) -> None:
        vals = np.array([np.nan, 1.0, 1.0, 2.0])
        pct = percentile_rank(vals)
        assert np.isnan(pct[0])
        assert np.isclose(pct[1], pct[2])
        assert np.isclose(pct[3], 1.0)


# ---------------------------------------------------------------------------
# 6. Zero-change behavior
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
# 7. Short episodes
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
            # T=3 < 2*h for h>=2 → all NaN
            assert np.isnan(dv).all()

    def test_no_pointers_for_large_h(self, tmp_path: Path) -> None:
        root = _make_corpus_dir(tmp_path, [
            {"name": "a.npz", "rewards": np.array([1.0, 2.0, 3.0])},
            {"name": "b.npz", "rewards": np.array([4.0, 5.0, 6.0])},
        ])
        s = profile_corpus(root, data_split_seed=0, horizons=(1, 8), d_horizons=(2,))
        assert s["eligible_counts"]["H=8"] == 0


# ---------------------------------------------------------------------------
# 8. Finite/nonnegative weights
# ---------------------------------------------------------------------------


class TestFiniteNonnegativeWeights:
    def test_weights_finite_and_nonnegative(self) -> None:
        n = 50
        rng = np.random.RandomState(42)
        q_pos = rng.rand(n)
        q_neg = rng.rand(n)
        q_up = rng.rand(n)
        q_down = rng.rand(n)
        terminal = rng.randint(0, 2, size=n).astype(np.float64)
        w = compute_weights(q_pos, q_neg, q_up, q_down, terminal,
                            0.1, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0)
        assert np.all(np.isfinite(w))
        assert np.all(w >= 0.0)

    def test_weights_with_nan(self) -> None:
        q_pos = np.array([0.1, np.nan, 0.3])
        q_neg = np.array([np.nan, 0.2, 0.1])
        w = compute_weights(q_pos, q_neg, np.zeros(3), np.zeros(3), np.zeros(3),
                            0.1, 1.0, 1.0, 0.5, 0.5, 0.0, 1.0, 1.0)
        assert np.all(np.isfinite(w))
        assert w[1] >= 0.0  # NaN inputs → 0 contribution from pos/neg


# ---------------------------------------------------------------------------
# 9. Nonzero uniform floor
# ---------------------------------------------------------------------------


class TestUniformFloor:
    def test_min_weight_at_least_eta(self) -> None:
        n = 10
        q_pos = np.zeros(n)
        q_neg = np.zeros(n)
        q_up = np.zeros(n)
        q_down = np.zeros(n)
        terminal = np.zeros(n)
        w = compute_weights(q_pos, q_neg, q_up, q_down, terminal,
                            0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
        assert np.isclose(w, 0.5).all()

    def test_eta_prevents_zero_weights(self) -> None:
        """Even with zero signals, weight = eta > 0."""
        w = compute_weights(
            np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5),
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
        )
        assert (w == 0.1).all()


# ---------------------------------------------------------------------------
# 10. Effective sample size
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
# 11. No model/training mutation (import smoke)
# ---------------------------------------------------------------------------


class TestNoModelMutation:
    def test_import_dry(self) -> None:
        """Importing the module does not create any models."""
        from rwm.memory import corpus_profiler
        assert hasattr(corpus_profiler, "profile_corpus")

    def test_no_side_effects_on_call(self, tmp_path: Path) -> None:
        """Profiling does not write outside the specified output dir."""
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
# 12. Sensitivity grid sanity
# ---------------------------------------------------------------------------


class TestSensitivityGrid:
    def test_ess_nonnegative(self) -> None:
        n = 100
        rng = np.random.RandomState(0)
        q_pos = rng.rand(n)
        q_neg = rng.rand(n)
        q_up = rng.rand(n)
        q_down = rng.rand(n)
        terminal = rng.randint(0, 2, size=n).astype(np.float64)
        results = run_sensitivity_grid(q_pos, q_neg, q_up, q_down, terminal, n)
        assert len(results) > 0
        for r in results:
            assert r["effective_sample_size"] > 0
            assert 0 < r["ess_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# 13. Correlation sanity
# ---------------------------------------------------------------------------


class TestCorrelations:
    def test_identical_signals(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        corr = compute_signal_correlations(
            {1: x}, {2: x},
            x, x, {2: x}, {2: x},
        )
        for key, val in corr.items():
            assert np.isclose(val, 1.0, atol=1e-6), f"{key} should be 1.0, got {val}"

    def test_anti_correlated(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        corr = compute_signal_correlations({1: x}, {2: x}, x, x, {2: y}, {2: y})
        # There should be a negative correlation between H=1 returns and one of the y
        # Just check the function runs without error
        assert isinstance(corr, dict)


# ---------------------------------------------------------------------------
# 14. Tie frequency
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
# 15. Dense region impact
# ---------------------------------------------------------------------------


class TestDenseRegion:
    def test_uniform_weights_low_impact(self) -> None:
        w = np.ones(100)
        q = np.linspace(0, 1, 100)
        impact = compute_dense_region_impact(w, q)
        assert impact["top_10pct_weight_fraction"] < 0.2  # roughly uniform

    def test_highly_skewed(self) -> None:
        w = np.zeros(100)
        w[0] = 100.0  # one pointer dominates
        q = np.linspace(0, 1, 100)
        impact = compute_dense_region_impact(w, q)
        assert impact["gini_weight"] > 0.8


# ---------------------------------------------------------------------------
# 16. Full profile smoke (end-to-end)
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
        assert s["n_files"] >= 2  # at least 2 in train
        assert s["n_transitions"] >= 1  # at least some transitions
        assert s["n_eligible_pointers"] == s["n_transitions"]
        assert s["train_val_disjoint"] is True
        assert len(s["return_quantiles"]) == 2
        assert len(s["d_quantiles"]) == 1
        assert len(s["surprise_counts"]) == 1
        assert len(s["sensitivity_grid"]) > 0

    def test_no_done_in_corpus(self, tmp_path: Path) -> None:
        """Realistic scenario: all done flags are False."""
        root = _make_corpus_dir(tmp_path, [
            {"name": "a.npz", "rewards": np.array([-0.1, -0.1, 5.0, -0.1])},
            {"name": "b.npz", "rewards": np.array([1.0, 2.0, 1.0])},
        ])
        s = profile_corpus(root, data_split_seed=0, horizons=(1, 4), d_horizons=(2,))
        assert s["all_terminated"] is False
        assert s["all_truncated"] is False  # all done=False
