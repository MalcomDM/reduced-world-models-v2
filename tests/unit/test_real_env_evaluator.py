"""Tests for Stage 5 real-environment evaluator.

Verifies:
  1. SRU step 0 receives temporal_state=None;
     every later SRU step receives the previous output.temporal_state;
     SRU never receives causal history/lengths.
  2. Causal step 0 receives history=None;
     every later causal step receives the previous output.history and lengths;
     causal never receives temporal_state.
  3. At environment step t, prev_action = action executed at t-1.
  4. Reward prediction uses the newly selected action_t before env.step(action_t).
  5. Actor actions remain within CarRacing bounds.
  6. Evaluator runs with no gradients, does not alter model params.
  7. Only dev seeds accepted; val/test seeds rejected.
  8. Deterministic mode gives identical action traces for same seed/model.
  9. CSV alignment: one predicted and one true reward per executed action.
 10. Zero-action baseline: all actions zero.
 11. Random baseline: actions bounded, deterministic, rng_seed persisted in metadata.
 12. Anchor integrity: hash mismatch raises error; legacy checkpoint warns.
 13. CLI mutual exclusion: --baseline and --random-baseline rejected together.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from rwm.config.config import ACTION_DIM, WORLD_STATE_DIM
from rwm.config.experiment_config import TemporalConfig
from rwm.evaluation.real_env_evaluator import (
    EpisodeResult,
    _validate_seed,
    compute_reward_mse_mae,
    mean_action,
    run_episode,
    run_random_baseline,
    run_zero_baseline,
    verify_actor_checkpoint_anchor,
)
from rwm.evaluation.schema import SeedManifest
from rwm.models.rwm.model import ReducedWorldModel
from rwm.trainers.imagined_actor_critic import (
    ImaginedACTrainer,
    ImaginedACTrainingConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
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


# ---------------------------------------------------------------------------
# Spy helper
# ---------------------------------------------------------------------------

class ModelForwardSpy:
    """Intercept model.forward, recording calls for analysis."""

    def __init__(self, model: ReducedWorldModel):
        self._orig_forward = model.forward
        self._model = model
        self.calls: List[Dict[str, Any]] = []
        model.forward = self._spy  # type: ignore[assignment]

    def _spy(self, *args, **kwargs) -> Any:
        output = self._orig_forward(*args, **kwargs)
        self.calls.append({
            "args": args,
            "kwargs": {k: v for k, v in kwargs.items()},
            "output": output,
        })
        return output

    def close(self) -> None:
        self._model.forward = self._orig_forward


def _make_sru_model_and_ac():
    """Return (model, ac) for MinimalSRU backend."""
    from torch.utils.data import DataLoader
    import tempfile

    tc = TemporalConfig(backend="minimal_sru")
    m = ReducedWorldModel(temporal_config=tc, tokenizer_eval_mode="mean").eval()
    for p in m.parameters():
        p.requires_grad_(False)

    loader = DataLoader(_FakeDataset(), batch_size=1, shuffle=False)
    cfg = ImaginedACTrainingConfig(warmup_steps=4, max_batches=1)
    tr = ImaginedACTrainer(model=m, train_loader=loader, train_cfg=cfg,
                           out_dir=Path(tempfile.mkdtemp()))
    return m, tr.ac


# ===================================================================
# 1. SRU state-carrying (spy-based)
# ===================================================================

class TestSRUStateCarrying:
    """Spy-based assertions for SRU backend."""

    @pytest.mark.envs
    def test_sru_step0_receives_temporal_state_none(self):
        m, ac = _make_sru_model_and_ac()
        spy = ModelForwardSpy(m)
        dev = SeedManifest(entries={"100": "dev"})
        ep = run_episode(m, ac, seed=100, manifest=dev, max_steps=5)
        spy.close()
        assert len(spy.calls) >= 1
        kw0 = spy.calls[0]["kwargs"]
        assert kw0.get("temporal_state") is None, "SRU step 0 must receive None temporal_state"
        assert kw0.get("history") is None, "SRU must never receive causal history at step 0"

    @pytest.mark.envs
    def test_sru_step1_receives_previous_temporal_state(self):
        m, ac = _make_sru_model_and_ac()
        spy = ModelForwardSpy(m)
        dev = SeedManifest(entries={"100": "dev"})
        run_episode(m, ac, seed=100, manifest=dev, max_steps=5)
        spy.close()
        assert len(spy.calls) >= 2
        assert spy.calls[0]["kwargs"].get("temporal_state") is None
        for i in range(1, len(spy.calls)):
            received = spy.calls[i]["kwargs"].get("temporal_state")
            expected = spy.calls[i - 1]["output"].temporal_state
            assert received is expected, (
                f"SRU step {i} did not receive the previous output state"
            )

    @pytest.mark.envs
    def test_sru_never_receives_causal_history_or_lengths(self):
        m, ac = _make_sru_model_and_ac()
        spy = ModelForwardSpy(m)
        dev = SeedManifest(entries={"100": "dev"})
        ep = run_episode(m, ac, seed=100, manifest=dev, max_steps=5)
        spy.close()
        for i, call in enumerate(spy.calls):
            kw = call["kwargs"]
            assert kw.get("history") is None, f"SRU step {i} received history"
            assert kw.get("lengths") is None, f"SRU step {i} received lengths"


# ===================================================================
# 2. Causal state-carrying (spy-based)
# ===================================================================

class TestCausalStateCarrying:
    """Spy-based assertions for causal backend."""

    @pytest.mark.envs
    def test_causal_step0_receives_history_none(self, model_and_ac, dev_manifest):
        m, ac = model_and_ac
        spy = ModelForwardSpy(m)
        ep = run_episode(m, ac, seed=100, manifest=dev_manifest, max_steps=5)
        spy.close()
        kw0 = spy.calls[0]["kwargs"]
        assert kw0.get("history") is None, "Causal step 0 must receive None history"
        assert kw0.get("temporal_state") is None, "Causal must never receive temporal_state"

    @pytest.mark.envs
    def test_causal_step1_receives_previous_history(self, model_and_ac, dev_manifest):
        m, ac = model_and_ac
        spy = ModelForwardSpy(m)
        run_episode(m, ac, seed=100, manifest=dev_manifest, max_steps=5)
        spy.close()
        assert len(spy.calls) >= 2
        assert spy.calls[0]["kwargs"].get("history") is None
        for i in range(1, len(spy.calls)):
            assert (
                spy.calls[i]["kwargs"].get("history")
                is spy.calls[i - 1]["output"].history
            ), f"Causal step {i} did not receive the previous history"
            assert (
                spy.calls[i]["kwargs"].get("lengths")
                is spy.calls[i - 1]["output"].lengths
            ), f"Causal step {i} did not receive the previous lengths"

    @pytest.mark.envs
    def test_causal_never_receives_temporal_state(self, model_and_ac, dev_manifest):
        m, ac = model_and_ac
        orig_forward = m.forward
        call_ts = []
        def spy(*args, **kwargs):
            call_ts.append(kwargs.get("temporal_state"))
            return orig_forward(*args, **kwargs)
        m.forward = spy  # type: ignore[assignment]
        ep = run_episode(m, ac, seed=100, manifest=dev_manifest, max_steps=5)
        m.forward = orig_forward
        for i, ts in enumerate(call_ts):
            assert ts is None, f"Causal step {i} must not receive temporal_state"


# ===================================================================
# 3. Action timing (spy-based)
# ===================================================================

class TestActionTiming:
    """prev_action matches previous executed action; reward uses newly chosen action."""

    @pytest.mark.envs
    def test_prev_action_matches_previous_step_executed_action(self, model_and_ac, dev_manifest):
        """At env step t, prev_action equals the action executed at t-1."""
        m, ac = model_and_ac
        orig_forward = m.forward
        observed_prev_actions = []
        def spy(*args, **kwargs):
            observed_prev_actions.append(kwargs.get("prev_action").squeeze(0).detach().cpu().numpy().copy())
            return orig_forward(*args, **kwargs)
        m.forward = spy  # type: ignore[assignment]
        ep = run_episode(m, ac, seed=100, manifest=dev_manifest, max_steps=5)
        m.forward = orig_forward
        assert len(observed_prev_actions) >= 2
        # prev_action[0] should be zeros (episode start)
        assert np.allclose(observed_prev_actions[0], np.zeros(3), atol=1e-6), \
            "First prev_action must be zeros"
        # prev_action[t] should equal the action executed at step t-1
        for t in range(1, len(observed_prev_actions)):
            prev_action_t = observed_prev_actions[t]
            step_t_minus_1 = ep.steps[t - 1]
            expected = np.array([step_t_minus_1["action_steer"],
                                 step_t_minus_1["action_gas"],
                                 step_t_minus_1["action_brake"]])
            assert np.allclose(prev_action_t, expected, atol=1e-5), \
                f"prev_action at step {t} must equal action executed at step {t-1}"

    @pytest.mark.envs
    def test_reward_prediction_uses_newly_chosen_action(self, model_and_ac, dev_manifest):
        """Reward prediction uses the action selected by Actor at step t, before env.step."""
        m, ac = model_and_ac
        # Capture the current_action passed to model.controller for reward prediction.
        orig_controller = m.controller.predict_reward
        reward_actions = []
        def spy_predict(h, a):
            reward_actions.append(a.detach().cpu().clone())
            return orig_controller(h, a)
        m.controller.predict_reward = spy_predict
        ep = run_episode(m, ac, seed=100, manifest=dev_manifest, max_steps=5)
        m.controller.predict_reward = orig_controller
        # Each environment step invokes the reward head first with the
        # placeholder previous action inside model.forward(), then with the
        # Actor's newly selected action. The second call must match env.step.
        assert len(reward_actions) == 2 * ep.n_steps
        for t, step in enumerate(ep.steps):
            expected = torch.tensor([
                step["action_steer"],
                step["action_gas"],
                step["action_brake"],
            ])
            torch.testing.assert_close(
                reward_actions[2 * t + 1].squeeze(0), expected,
            )


# ===================================================================
# 4. Action bounds
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
# 5. No gradients, no param change
# ===================================================================

class TestNoGradients:
    @pytest.mark.envs
    def test_no_gradients_after_eval(self, model_and_ac, dev_manifest):
        model, ac = model_and_ac
        snap = {k: v.data.clone() for k, v in model.state_dict().items()}
        _ = run_episode(model, ac, seed=100, manifest=dev_manifest, max_steps=5)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, snap[k], msg=f"{k} changed")
        for p in model.parameters():
            assert p.grad is None, f"gradient leaked to {p.shape}"
        for p in ac.actor.parameters():
            assert p.grad is None, f"gradient leaked to actor {p.shape}"


# ===================================================================
# 6. Seed validation
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
# 7. Deterministic reproducibility
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
# 8. CSV alignment
# ===================================================================

class TestCSVAlignment:
    @pytest.mark.envs
    def test_one_reward_per_action(self, model_and_ac, dev_manifest):
        model, ac = model_and_ac
        ep = run_episode(model, ac, seed=100, manifest=dev_manifest, max_steps=10)
        assert len(ep.steps) == ep.n_steps
        for i, s in enumerate(ep.steps):
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
        assert len(lines) == ep.n_steps + 1
        assert "reward_true" in lines[0]
        assert "reward_pred" in lines[0]


# ===================================================================
# 9. Zero-action baseline
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
# 10. Random baseline
# ===================================================================

class TestRandomBaseline:
    @pytest.mark.envs
    def test_random_baseline_runs(self):
        dev = SeedManifest(entries={"100": "dev", "101": "dev"})
        ep = run_random_baseline(seed=100, manifest=dev, max_steps=5)
        assert ep.n_steps > 0
        for s in ep.steps:
            assert -1.0 <= s["action_steer"] <= 1.0
            assert 0.0 <= s["action_gas"] <= 1.0
            assert 0.0 <= s["action_brake"] <= 1.0

    @pytest.mark.envs
    def test_random_baseline_deterministic(self):
        dev = SeedManifest(entries={"100": "dev"})
        ep1 = run_random_baseline(seed=100, manifest=dev, max_steps=5, rng_seed=42)
        ep2 = run_random_baseline(seed=100, manifest=dev, max_steps=5, rng_seed=42)
        assert ep1.n_steps == ep2.n_steps
        for s1, s2 in zip(ep1.steps, ep2.steps):
            assert s1["action_steer"] == s2["action_steer"]
            assert s1["action_gas"] == s2["action_gas"]
            assert s1["action_brake"] == s2["action_brake"]

    @pytest.mark.envs
    def test_random_baseline_differs_by_seed(self):
        dev = SeedManifest(entries={"100": "dev", "101": "dev"})
        ep1 = run_random_baseline(seed=100, manifest=dev, max_steps=5, rng_seed=42)
        ep2 = run_random_baseline(seed=101, manifest=dev, max_steps=5, rng_seed=99)
        if ep1.n_steps > 0 and ep2.n_steps > 0:
            all_same = all(
                s1["action_steer"] == s2["action_steer"]
                for s1, s2 in zip(ep1.steps, ep2.steps)
            )
            assert not all_same, "Different seeds should produce different actions"

    @pytest.mark.envs
    def test_random_baseline_default_rng_seed(self):
        """Default rng_seed = env_seed + 10000, persisted in episode metadata."""
        dev = SeedManifest(entries={"100": "dev"})
        ep = run_random_baseline(seed=100, manifest=dev, max_steps=3)
        assert ep.rng_seed == 10100, f"Expected rng_seed=10100, got {ep.rng_seed}"

    @pytest.mark.envs
    def test_random_baseline_json_persists_rng_seed(self):
        """Episode JSON output includes rng_seed."""
        dev = SeedManifest(entries={"100": "dev"})
        ep = run_random_baseline(seed=100, manifest=dev, max_steps=3, rng_seed=42)
        tmp = Path(tempfile.mkdtemp()) / "test.json"
        from rwm.evaluation.real_env_evaluator import save_episode_json
        save_episode_json(ep, tmp)
        data = json.loads(tmp.read_text())
        assert data.get("rng_seed") == 42, f"JSON must contain rng_seed, got {data}"

    @pytest.mark.envs
    def test_random_baseline_reproduces_with_saved_rng_seed(self):
        """Re-running with the persisted rng_seed gives identical trace."""
        dev = SeedManifest(entries={"100": "dev"})
        ep1 = run_random_baseline(seed=100, manifest=dev, max_steps=5)
        saved_rng = ep1.rng_seed
        ep2 = run_random_baseline(seed=100, manifest=dev, max_steps=5,
                                  rng_seed=saved_rng)
        assert ep1.n_steps == ep2.n_steps
        for s1, s2 in zip(ep1.steps, ep2.steps):
            assert s1["action_steer"] == s2["action_steer"]
            assert s1["action_gas"] == s2["action_gas"]
            assert s1["action_brake"] == s2["action_brake"]


# ===================================================================
# 11. Anchor integrity
# ===================================================================

class TestAnchorIntegrity:
    """Matching / mismatching / legacy anchor hashes."""

    def _save_fake_ac_checkpoint(self, path: Path, anchor_hash: Optional[str] = None):
        """Save a minimal AC checkpoint with optional anchor info."""
        import torch
        data = {
            "schema_version": 1,
            "kind": "imagined_actor_critic",
            "step": 1,
            "global_step": 1,
            "actor_critic": {"actor": {}, "critic": {}, "target_critic": {}},
            "optimizer": {"actor_optim": {}, "critic_optim": {}},
            "config": {},
            "actor_critic_config": {},
        }
        if anchor_hash is not None:
            data["anchor"] = {"path": "/fake/path", "hash": anchor_hash}
        torch.save(data, path)

    def _save_fake_anchor(self, path: Path) -> str:
        """Save a minimal anchor checkpoint and return its hash."""
        import torch, hashlib
        torch.save({"model_state": {}, "config": {}}, path)
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()[:16]

    def test_matching_hashes_proceed(self, tmp_path: Path):
        """Matching anchor hash does not raise."""
        anchor_path = tmp_path / "anchor.pt"
        ac_path = tmp_path / "ac.pt"
        anchor_hash = self._save_fake_anchor(anchor_path)
        self._save_fake_ac_checkpoint(ac_path, anchor_hash=anchor_hash)
        actual, verified = verify_actor_checkpoint_anchor(
            anchor_path, ac_path,
        )
        assert actual == anchor_hash
        assert verified is True

    def test_mismatching_hashes_exit(self, tmp_path: Path):
        """Mismatching anchor hash raises SystemExit."""
        anchor_path = tmp_path / "anchor.pt"
        ac_path = tmp_path / "ac.pt"
        self._save_fake_anchor(anchor_path)
        self._save_fake_ac_checkpoint(ac_path, anchor_hash="aaaaaaaaaaaaaaaa")
        with pytest.raises(ValueError, match="Anchor hash mismatch"):
            verify_actor_checkpoint_anchor(anchor_path, ac_path)

    def test_legacy_checkpoint_warns(self, tmp_path: Path):
        """Legacy checkpoint without anchor hash prints warning."""
        anchor_path = tmp_path / "anchor.pt"
        ac_path = tmp_path / "ac.pt"
        self._save_fake_anchor(anchor_path)
        self._save_fake_ac_checkpoint(ac_path, anchor_hash=None)
        with pytest.warns(RuntimeWarning, match="cannot be verified"):
            _, verified = verify_actor_checkpoint_anchor(
                anchor_path, ac_path,
            )
        assert verified is False


# ===================================================================
# 12. CLI mutual exclusion
# ===================================================================

class TestCLIMutualExclusion:
    """--baseline and --random-baseline must be rejected together."""

    def test_both_baselines_rejected(self):
        """Exercise the production CLI parser, not a duplicate test parser."""
        script = (
            Path(__file__).parents[2]
            / "scripts/evaluation/evaluate_real_env.py"
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--baseline",
                "--random-baseline",
            ],
            cwd=Path(__file__).parents[2],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode != 0
        assert "mutually exclusive" in completed.stderr


# ===================================================================
# 13. Helper functions
# ===================================================================

class TestHelpers:
    def test_compute_reward_mse_mae(self):
        ep = EpisodeResult(seed=0)
        ep.record_step(np.zeros(3), 1.0, 0.5, 0.0, np.zeros(3))
        ep.record_step(np.zeros(3), 2.0, 1.5, 0.0, np.zeros(3))
        mse, mae = compute_reward_mse_mae(ep)
        assert mse == pytest.approx(0.25)
        assert mae == pytest.approx(0.5)

    def test_mean_action(self):
        ep = EpisodeResult(seed=0)
        ep.record_step(np.array([1.0, 0.5, 0.0]), 0.0, 0.0, 0.0, np.zeros(3))
        ep.record_step(np.array([-1.0, 0.0, 1.0]), 0.0, 0.0, 0.0, np.zeros(3))
        ma = mean_action(ep)
        assert ma[0] == pytest.approx(0.0)
        assert ma[1] == pytest.approx(0.25)
        assert ma[2] == pytest.approx(0.5)
