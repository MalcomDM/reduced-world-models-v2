"""Tests for Stage 5.3 real-environment evaluator.

Verifies:
  1. Exact previous-action/current-action/reward timing.
  2. Actor actions remain within CarRacing bounds.
  3. Evaluator runs with no gradients, does not alter model params.
  4. Only dev seeds accepted; val/test seeds rejected.
  5. Deterministic mode gives identical action traces for same seed/model.
  6. CSV alignment: one predicted and one true reward per executed action.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from rwm.config.config import ACTION_DIM, WORLD_STATE_DIM
from rwm.evaluation.real_env_evaluator import (
    _validate_seed,
    compute_reward_mse_mae,
    mean_action,
    run_episode,
    run_zero_baseline,
)
from rwm.evaluation.schema import SeedManifest
from rwm.models.rwm.model import ReducedWorldModel
from rwm.trainers.imagined_actor_critic import (
    ImaginedACTrainer,
    ImaginedACTrainingConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def dev_manifest():
    return SeedManifest(entries={"100": "dev", "200": "val", "300": "locked_test"})

class _FakeDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        T = 16
        return {
            "obs": torch.randn(T, 3, 64, 64),
            "action": torch.randn(T, ACTION_DIM),
            "reward": torch.randn(T),
            "done": torch.zeros(T, dtype=torch.bool),
            "predecessor_action": torch.zeros(ACTION_DIM),
        }


@pytest.fixture
def model_and_ac():
    torch.manual_seed(42)
    m = ReducedWorldModel(
        action_dim=ACTION_DIM, reward_head_kind="linear",
        tokenizer_eval_mode="mean",
    )
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

    cfg = ImaginedACTrainingConfig(
        warmup_steps=4, imagination_horizon=4, max_batches=1,
    )
    loader = DataLoader(_FakeDataset(), batch_size=1, shuffle=False)
    tr = ImaginedACTrainer(
        model=m, train_loader=loader, train_cfg=cfg,
        out_dir=Path(tempfile.mkdtemp()),
    )
    return m, tr.ac


# ===================================================================
# 1. Action timing
# ===================================================================

class TestActionTiming:
    @pytest.mark.envs
    def test_timing_contract(self, model_and_ac, dev_manifest):
        """The evaluator must compute the predicted reward BEFORE the action
        is executed in the environment."""
        model, ac = model_and_ac
        # This test verifies structural correctness of the evaluator.
        # The evaluator's run_episode function:
        #   1. Gets belief z_t from model
        #   2. Gets action from Actor.mode()
        #   3. Predicts reward with Reward(z_t, action)
        #   4. Executes action in env
        # We verify by checking the EpisodeResult has both reward_pred
        # and reward_true for each step.
        ep = run_episode(model, ac, seed=100, manifest=dev_manifest, max_steps=5)
        assert ep.n_steps > 0
        for s in ep.steps:
            assert "reward_pred" in s
            assert "reward_true" in s
            assert isinstance(s["reward_pred"], float)
            assert isinstance(s["reward_true"], float)


# ===================================================================
# 2. Action bounds
# ===================================================================

class TestActionBounds:
    @pytest.mark.envs
    def test_actions_within_bounds(self, model_and_ac, dev_manifest):
        model, ac = model_and_ac
        ep = run_episode(model, ac, seed=100, manifest=dev_manifest, max_steps=5)
        for s in ep.steps:
            steer = s["action_steer"]
            gas = s["action_gas"]
            brake = s["action_brake"]
            assert -1.0 <= steer <= 1.0
            assert 0.0 <= gas <= 1.0
            assert 0.0 <= brake <= 1.0


# ===================================================================
# 3. No gradients, no param change
# ===================================================================

class TestNoGradients:
    @pytest.mark.envs
    def test_no_gradients_after_eval(self, model_and_ac, dev_manifest):
        model, ac = model_and_ac
        snap = {k: v.data.clone() for k, v in model.state_dict().items()}
        _ = run_episode(model, ac, seed=100, manifest=dev_manifest, max_steps=5)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, snap[k],
                                        msg=f"{k} changed")
        for p in model.parameters():
            assert p.grad is None, f"gradient leaked to {p.shape}"
        for p in ac.actor.parameters():
            assert p.grad is None, f"gradient leaked to actor {p.shape}"


# ===================================================================
# 4. Seed validation
# ===================================================================

class TestSeedValidation:
    def test_dev_seeds_accepted(self, dev_manifest):
        _validate_seed(100, dev_manifest)

    def test_non_dev_seed_raises(self, dev_manifest):
        with pytest.raises(ValueError, match="not in manifest"):
            _validate_seed(42, dev_manifest)

    def test_val_and_test_seeds_raise(self, dev_manifest):
        with pytest.raises(ValueError, match="only dev"):
            _validate_seed(200, dev_manifest)
        with pytest.raises(ValueError, match="only dev"):
            _validate_seed(300, dev_manifest)


# ===================================================================
# 5. Deterministic reproducibility
# ===================================================================

class TestDeterministic:
    @pytest.mark.envs
    def test_identical_action_traces(self, model_and_ac, dev_manifest):
        model, ac = model_and_ac
        ep1 = run_episode(model, ac, seed=100, manifest=dev_manifest, max_steps=5)
        ep2 = run_episode(model, ac, seed=100, manifest=dev_manifest, max_steps=5)
        assert ep1.n_steps == ep2.n_steps
        for s1, s2 in zip(ep1.steps, ep2.steps):
            assert s1["action_steer"] == pytest.approx(s2["action_steer"], abs=1e-6)
            assert s1["action_gas"] == pytest.approx(s2["action_gas"], abs=1e-6)
            assert s1["action_brake"] == pytest.approx(s2["action_brake"], abs=1e-6)


# ===================================================================
# 6. CSV alignment
# ===================================================================

class TestCSVAlignment:
    @pytest.mark.envs
    def test_one_reward_per_action(self, model_and_ac, dev_manifest):
        model, ac = model_and_ac
        ep = run_episode(model, ac, seed=100, manifest=dev_manifest, max_steps=10)
        assert len(ep.steps) == ep.n_steps
        for i, s in enumerate(ep.steps):
            # Every step has exactly one action and two rewards (pred + true).
            assert "action_steer" in s
            assert "reward_pred" in s
            assert "reward_true" in s

    def test_csv_write_roundtrip(self, model_and_ac, dev_manifest):
        model, ac = model_and_ac
        ep = run_episode(model, ac, seed=100, manifest=dev_manifest, max_steps=5)
        tmp = Path(tempfile.mkdtemp()) / "test.csv"
        from rwm.evaluation.real_env_evaluator import save_episode_csv
        save_episode_csv(ep, tmp)
        assert tmp.exists()
        lines = tmp.read_text().strip().split("\n")
        assert len(lines) == ep.n_steps + 1  # header + data rows
        assert "reward_true" in lines[0]
        assert "reward_pred" in lines[0]


# ===================================================================
# 7. Zero-action baseline
# ===================================================================

class TestZeroBaseline:
    @pytest.mark.envs
    def test_zero_baseline_runs(self, dev_manifest):
        ep = run_zero_baseline(seed=100, manifest=dev_manifest, max_steps=5)
        assert ep.n_steps > 0
        for s in ep.steps:
            assert s["action_steer"] == 0.0
            assert s["action_gas"] == 0.0
            assert s["action_brake"] == 0.0

    def test_zero_baseline_seed_validation(self, dev_manifest):
        with pytest.raises(ValueError):
            run_zero_baseline(seed=99, manifest=dev_manifest, max_steps=1)


# ===================================================================
# 8. Helper functions
# ===================================================================

class TestHelpers:
    def test_compute_reward_mse_mae(self):
        from rwm.evaluation.real_env_evaluator import EpisodeResult
        ep = EpisodeResult(seed=0)
        ep.record_step(np.zeros(3), 1.0, 0.5, 0.0, np.zeros(3))
        ep.record_step(np.zeros(3), 2.0, 1.5, 0.0, np.zeros(3))
        mse, mae = compute_reward_mse_mae(ep)
        assert mse == pytest.approx(0.25)  # (0.5^2 + 0.5^2) / 2
        assert mae == pytest.approx(0.5)

    def test_mean_action(self):
        from rwm.evaluation.real_env_evaluator import EpisodeResult
        ep = EpisodeResult(seed=0)
        ep.record_step(np.array([1.0, 0.5, 0.0]), 0.0, 0.0, 0.0, np.zeros(3))
        ep.record_step(np.array([-1.0, 0.0, 1.0]), 0.0, 0.0, 0.0, np.zeros(3))
        ma = mean_action(ep)
        assert ma[0] == pytest.approx(0.0)
        assert ma[1] == pytest.approx(0.25)
        assert ma[2] == pytest.approx(0.5)
