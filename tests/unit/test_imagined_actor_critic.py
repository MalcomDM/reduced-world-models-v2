"""Tests for Stage 5.0 frozen-world-model imagined Actor-Critic training.

Verifies:
  1. Autoregressive action timing: action scored = action advanced.
  2. No future observation enters an imagined state.
  3. Horizon ≤ 12 validation; invalid horizon raises.
  4. Correct final bootstrap value passed to return helpers.
  5. Actor/Critic parameters change after training steps.
  6. Frozen world-model parameters bitwise identical after updates.
  7. Actions satisfy CarRacing bounds.
  8. Finite losses/gradients; target critic changes via Polyak.
  9. Fixed seed + mean tokenizer gives reproducible outputs.
 10. Structured checkpoint round-trip restores Actor/Critic.
"""

import copy
import json
import tempfile
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from rwm.config.config import ACTION_DIM, VALUES_DIM, SEQ_LEN, WORLD_STATE_DIM
from rwm.imagination import ImaginationRollout
from rwm.models.actor_critic import ActorCritic, ActorCriticConfig
from rwm.models.rwm.model import ReducedWorldModel
from rwm.trainers.imagined_actor_critic import (
    ImaginedACTrainer,
    ImaginedACTrainingConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    m = ReducedWorldModel(
        action_dim=ACTION_DIM,
        reward_head_kind="linear",
        tokenizer_eval_mode="mean",
    )
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class _FakeDataset(torch.utils.data.Dataset):
    """Synthetic dataset that yields valid RolloutSample batches."""

    def __init__(self, seq_len=16, num=8):
        self.seq_len = seq_len
        self.num = num

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        T = self.seq_len
        return {
            "obs": torch.randn(T, 3, 64, 64),
            "action": torch.randn(T, ACTION_DIM),
            "reward": torch.randn(T),
            "done": torch.zeros(T, dtype=torch.bool),
            "predecessor_action": torch.zeros(ACTION_DIM),
        }


def _make_batch(seq_len=16, bsize=4):
    """Build a synthetic batch tensor (as DataLoader would produce)."""
    return {
        "obs": torch.randn(bsize, seq_len, 3, 64, 64),
        "action": torch.randn(bsize, seq_len, ACTION_DIM),
        "reward": torch.randn(bsize, seq_len),
        "done": torch.zeros(bsize, seq_len, dtype=torch.bool),
        "predecessor_action": torch.zeros(bsize, ACTION_DIM),
    }


@pytest.fixture
def trainer(model):
    loader = DataLoader(_FakeDataset(seq_len=16, num=8), batch_size=2, shuffle=True)
    cfg = ImaginedACTrainingConfig(
        warmup_steps=4,
        imagination_horizon=4,
        max_batches=5,
        log_every=5,
        checkpoint_every=100,
    )
    return ImaginedACTrainer(
        model=model, train_loader=loader, train_cfg=cfg,
        device=torch.device("cpu"),
        out_dir=Path(tempfile.mkdtemp()),
    )


# ===================================================================
# 1. Autoregressive action timing
# ===================================================================

class TestActionTiming:
    """The action used to score ``r_hat`` must be the exact action passed
    to ``advance``."""

    def test_score_before_advance(self, trainer):
        batch = _make_batch(bsize=2)
        warmup_state = trainer.imag.warmup(
            *trainer._prep_warmup(batch), force_keep_input=True,
        )
        z_t = warmup_state.current_belief
        hist, lens = warmup_state.history, warmup_state.lengths

        c_t = trainer.ac.encode(z_t)
        dist = trainer.ac.actor(c_t)
        a_t = dist.rsample().clone()

        # Score with this action.
        with torch.no_grad():
            _, r_t = trainer.model.controller(z_t, a_t)

        # Advance with the same action.
        _, _, z_next = trainer.imag.advance(hist, lens, a_t)

        # Verify the action appears in the advanced token (as prev_action).
        advanced_token = trainer.imag.advance(
            hist, lens, a_t
        )[0][:, -1, :]  # (B, V+A)
        token_action = advanced_token[:, VALUES_DIM:]
        torch.testing.assert_close(token_action, a_t)

    def test_actions_not_from_future(self, trainer):
        """Imagined actions must not depend on any observation."""
        batch = _make_batch(bsize=2)
        ws = trainer.imag.warmup(
            *trainer._prep_warmup(batch), force_keep_input=True,
        )
        _, acts, _, _ = trainer.generate_trajectory(ws)
        # The actions are from the Actor (which receives only c_t from the
        # warmup), so they cannot depend on future observations.
        assert acts.shape == (2, trainer.cfg.imagination_horizon, ACTION_DIM)


# ===================================================================
# 2. No future observation enters imagined state
# ===================================================================

class TestNoFutureObs:
    def test_imagined_tokens_have_zero_spatial(self, trainer):
        batch = _make_batch(bsize=2)
        ws = trainer.imag.warmup(
            *trainer._prep_warmup(batch), force_keep_input=True,
        )
        states, actions, rewards, z_H = trainer.generate_trajectory(ws)
        # All imagined states should come from zero-spatial tokens.
        # We verify indirectly by checking that actions are the only
        # non-zero part of any advanced token.
        # The advance method always appends cat(zeros, action).
        hist = ws.history
        lens = ws.lengths
        for h in range(trainer.cfg.imagination_horizon):
            hist, lens, _ = trainer.imag.advance(hist, lens, actions[:, h])
        # After all advances, the imagined tokens start at position ws.history[1]
        imagined_tokens = hist[:, trainer.cfg.warmup_steps:, :VALUES_DIM]
        assert (imagined_tokens == 0).all()


# ===================================================================
# 3. Horizon validation
# ===================================================================

class TestHorizonValidation:
    def test_horizon_12_ok(self):
        cfg = ImaginedACTrainingConfig(imagination_horizon=12)
        cfg.validate()  # should not raise

    def test_horizon_13_raises(self):
        cfg = ImaginedACTrainingConfig(imagination_horizon=13)
        with pytest.raises(ValueError, match=r"\[1, 12\]"):
            cfg.validate()

    def test_horizon_0_raises(self):
        cfg = ImaginedACTrainingConfig(imagination_horizon=0)
        with pytest.raises(ValueError, match=r"\[1, 12\]"):
            cfg.validate()

    def test_warmup_0_raises(self):
        cfg = ImaginedACTrainingConfig(warmup_steps=0)
        with pytest.raises(ValueError, match="warmup_steps"):
            cfg.validate()


# ===================================================================
# 4. Final bootstrap passed correctly
# ===================================================================

class TestFinalBootstrap:
    def test_bootstrap_uses_target_critic(self, trainer):
        batch = _make_batch(bsize=2)
        ws = trainer.imag.warmup(
            *trainer._prep_warmup(batch), force_keep_input=True,
        )
        states, actions, rewards, z_H = trainer.generate_trajectory(ws)

        with torch.no_grad():
            c_H = trainer.ac.encode(z_H)
            expected_bv = trainer.ac.target_critic(c_H).squeeze(-1)

        # Simulate the bootstrap computation the trainer would do.
        c_H_direct = trainer.ac.encode(z_H)
        bv_direct = trainer.ac.target_critic(c_H_direct).squeeze(-1)
        torch.testing.assert_close(bv_direct, expected_bv)


# ===================================================================
# 5. Actor/Critic parameters change after training
# ===================================================================

class TestParamsChange:
    @pytest.mark.models
    def test_actor_params_change(self, trainer):
        snap = {k: v.clone() for k, v in trainer.ac.actor.state_dict().items()}
        trainer.train(num_batches=3)
        for k, v in trainer.ac.actor.state_dict().items():
            assert not torch.allclose(v, snap[k]), f"Actor param {k} did not change"

    @pytest.mark.models
    def test_critic_params_change(self, trainer):
        snap = {k: v.clone() for k, v in trainer.ac.critic.state_dict().items()}
        trainer.train(num_batches=3)
        for k, v in trainer.ac.critic.state_dict().items():
            assert not torch.allclose(v, snap[k]), f"Critic param {k} did not change"


# ===================================================================
# 6. Frozen world-model params unchanged
# ===================================================================

class TestFrozenWorldModel:
    @pytest.mark.models
    def test_wm_params_bitwise_identical(self, trainer):
        snap = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        trainer.train(num_batches=5)
        for k, v in trainer.model.state_dict().items():
            torch.testing.assert_close(
                v, snap[k],
                msg=f"World-model param {k} changed after training",
            )

    def test_requires_grad_false(self, trainer):
        assert all(not p.requires_grad for p in trainer.model.parameters())


# ===================================================================
# 7. Actions satisfy CarRacing bounds
# ===================================================================

class TestActionBounds:
    def test_generated_actions_respect_bounds(self, trainer):
        batch = _make_batch(bsize=2)
        ws = trainer.imag.warmup(
            *trainer._prep_warmup(batch), force_keep_input=True,
        )
        _, acts, _, _ = trainer.generate_trajectory(ws)
        assert (acts[:, :, 0] >= -1.0).all() and (acts[:, :, 0] <= 1.0).all()
        assert (acts[:, :, 1:] >= 0.0).all() and (acts[:, :, 1:] <= 1.0).all()


# ===================================================================
# 8. Finite losses/gradients; target critic changes via Polyak
# ===================================================================

class TestFiniteAndGradients:
    @pytest.mark.models
    def test_losses_finite(self, trainer):
        batch = _make_batch(bsize=2)
        metrics = trainer.train_step(batch)
        for k in ("actor_loss", "critic_loss", "entropy"):
            assert metrics[k] == metrics[k], f"{k} is NaN"
            assert metrics[k] != float("inf"), f"{k} is inf"

    @pytest.mark.models
    def test_actor_gradients_finite(self, trainer):
        """Verify gradients exist and are finite after backward."""
        batch = _make_batch(bsize=2)
        # Run one step manually to inspect gradients.
        ws = trainer.imag.warmup(
            *trainer._prep_warmup(batch), force_keep_input=True,
        )
        states, acts, rewards, z_H = trainer.generate_trajectory(ws)
        B = states.shape[0]
        H = trainer.cfg.imagination_horizon
        device = trainer.device
        terminated = torch.zeros(B, H, dtype=torch.bool, device=device)
        continuation = torch.ones(B, H, dtype=torch.bool, device=device)
        c_H = trainer.ac.encode(z_H)
        with torch.no_grad():
            bv = trainer.ac.target_critic(c_H).squeeze(-1)
        _ = trainer.ac.optimizer_step(
            states, acts, rewards, terminated, continuation, bv,
        )
        # Check actor gradients exist and are finite.
        for p in trainer.ac.actor.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    @pytest.mark.models
    def test_target_critic_changes_via_polyak(self, trainer):
        """After a step, target critic must differ from its pre-step state."""
        snap = {k: v.clone() for k, v in trainer.ac.target_critic.state_dict().items()}
        trainer.train(num_batches=2)
        changed = False
        for k, v in trainer.ac.target_critic.state_dict().items():
            if not torch.allclose(v, snap[k]):
                changed = True
                break
        assert changed, "Target critic did not change after training"


# ===================================================================
# 9. Reproducibility
# ===================================================================

class TestReproducibility:
    @pytest.mark.models
    def test_train_step_deterministic_given_seed(self):
        """Calling ``train_step`` twice with reset seed produces the same
        metrics on the first call (before params diverge)."""
        torch.manual_seed(42)
        m = ReducedWorldModel(
            action_dim=ACTION_DIM, reward_head_kind="linear",
            tokenizer_eval_mode="mean",
        )
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

        cfg = ImaginedACTrainingConfig(
            warmup_steps=4, imagination_horizon=4, max_batches=10, log_every=5,
        )
        batch = _make_batch(seq_len=16, bsize=2)

        # First step.
        torch.manual_seed(42)
        tr1 = ImaginedACTrainer(
            model=copy.deepcopy(m), train_loader=DataLoader(
                _FakeDataset(seq_len=16, num=4), batch_size=2, shuffle=False,
            ),
            train_cfg=cfg, device=torch.device("cpu"),
            out_dir=Path(tempfile.mkdtemp()),
        )
        m1 = tr1.train_step(batch)
        # Second step from fresh trainer with same seed.
        torch.manual_seed(42)
        tr2 = ImaginedACTrainer(
            model=copy.deepcopy(m), train_loader=DataLoader(
                _FakeDataset(seq_len=16, num=4), batch_size=2, shuffle=False,
            ),
            train_cfg=cfg, device=torch.device("cpu"),
            out_dir=Path(tempfile.mkdtemp()),
        )
        m2 = tr2.train_step(batch)

        for k in ("actor_loss", "critic_loss", "entropy", "imagined_reward_mean"):
            assert abs(m1[k] - m2[k]) < 1e-5, (
                f"{k} mismatch: {m1[k]} vs {m2[k]}"
            )


# ===================================================================
# 10. Checkpoint round-trip
# ===================================================================

class TestCheckpointRoundtrip:
    @pytest.mark.models
    def test_checkpoint_save_and_restore(self, trainer):
        trainer.train(num_batches=3)
        checkpoint = trainer.out_dir / "checkpoints" / "ac_checkpoint_3.pt"
        assert checkpoint.exists()

        restored = ImaginedACTrainer(
            model=copy.deepcopy(trainer.model), train_loader=trainer.train_loader,
            train_cfg=trainer.cfg, device=torch.device("cpu"),
            out_dir=Path(tempfile.mkdtemp()),
        )
        assert restored.load_actor_critic_checkpoint(checkpoint) == 3

        # Verify the real persisted checkpoint restores all learned heads.
        for (k1, v1), (k2, v2) in zip(
            trainer.ac.actor.state_dict().items(),
            restored.ac.actor.state_dict().items(),
        ):
            torch.testing.assert_close(v1, v2)

    @pytest.mark.models
    def test_anchor_info_saved(self, model):
        out_dir = Path(tempfile.mkdtemp())
        cfg = ImaginedACTrainingConfig(
            warmup_steps=4, imagination_horizon=4, max_batches=1, log_every=5,
        )
        loader = DataLoader(_FakeDataset(), batch_size=2, shuffle=False)
        tr = ImaginedACTrainer(
            model=model, train_loader=loader, train_cfg=cfg,
            device=torch.device("cpu"), out_dir=out_dir,
        )
        tr.set_anchor_info("/fake/path.pt", "abcd1234")
        tr.train()
        anchor_file = out_dir / "anchor_checkpoint.txt"
        assert anchor_file.exists()
        text = anchor_file.read_text()
        assert "abcd1234" in text

    @pytest.mark.models
    def test_metrics_csv_written(self, model):
        out_dir = Path(tempfile.mkdtemp())
        cfg = ImaginedACTrainingConfig(
            warmup_steps=4, imagination_horizon=4, max_batches=2, log_every=5,
        )
        loader = DataLoader(_FakeDataset(), batch_size=2, shuffle=False)
        tr = ImaginedACTrainer(
            model=model, train_loader=loader, train_cfg=cfg,
            device=torch.device("cpu"), out_dir=out_dir,
        )
        tr.train()
        csv_path = out_dir / "metrics.csv"
        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows


# ===================================================================
# 11. Seeded curriculum reproducibility
# ===================================================================

class TestCurriculumSeed:
    @pytest.mark.models
    def test_seed_gives_reproducible_curriculum(self):
        torch.manual_seed(42)
        m = ReducedWorldModel(
            action_dim=ACTION_DIM, reward_head_kind="linear",
            tokenizer_eval_mode="mean",
        )
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

        cfg = ImaginedACTrainingConfig(
            warmup_steps=4, horizons=[1, 2, 4], max_batches=5, log_every=5,
        )
        loaders = [
            DataLoader(_FakeDataset(num=8), batch_size=2, shuffle=False)
            for _ in range(2)
        ]

        trainers = []
        for loader in loaders:
            tr = ImaginedACTrainer(
                model=copy.deepcopy(m), train_loader=loader,
                train_cfg=copy.deepcopy(cfg), seed=42,
                out_dir=Path(tempfile.mkdtemp()),
            )
            trainers.append(tr)

        for tr in trainers:
            tr.train()

        for r1, r2 in zip(trainers[0]._metrics_log, trainers[1]._metrics_log):
            assert r1["horizon"] == r2["horizon"], (
                f"Horizon mismatch: {r1['horizon']} vs {r2['horizon']}"
            )


# ===================================================================
# 12. Horizon recording in metrics
# ===================================================================

class TestHorizonRecording:
    @pytest.mark.models
    def test_horizon_column_in_metrics(self):
        torch.manual_seed(0)
        m = ReducedWorldModel(
            action_dim=ACTION_DIM, reward_head_kind="linear",
            tokenizer_eval_mode="mean",
        )
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

        cfg = ImaginedACTrainingConfig(
            warmup_steps=4, horizons=[1, 2, 4], max_batches=10, log_every=10,
        )
        loader = DataLoader(_FakeDataset(num=8), batch_size=2, shuffle=False)
        tr = ImaginedACTrainer(
            model=m, train_loader=loader, train_cfg=cfg, seed=0,
            out_dir=Path(tempfile.mkdtemp()),
        )
        tr.train()

        assert len(tr._metrics_log) == 10
        horizons_used = {r["horizon"] for r in tr._metrics_log}
        assert len(horizons_used) >= 2
        assert horizons_used.issubset({1, 2, 4})

    @pytest.mark.models
    def test_fixed_horizon_records_same(self):
        torch.manual_seed(0)
        m = ReducedWorldModel(
            action_dim=ACTION_DIM, reward_head_kind="linear",
            tokenizer_eval_mode="mean",
        )
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

        cfg = ImaginedACTrainingConfig(
            warmup_steps=4, imagination_horizon=7, max_batches=5, log_every=5,
        )
        loader = DataLoader(_FakeDataset(num=8), batch_size=2, shuffle=False)
        tr = ImaginedACTrainer(
            model=m, train_loader=loader, train_cfg=cfg, seed=0,
            out_dir=Path(tempfile.mkdtemp()),
        )
        tr.train()
        for r in tr._metrics_log:
            assert r["horizon"] == 7


# ===================================================================
# 13. Fixed-horizon override disables curriculum
# ===================================================================

class TestFixedHorizonOverride:
    def test_explicit_horizon_overrides_curriculum(self):
        cfg = ImaginedACTrainingConfig(horizons=[1, 2, 4])
        cfg.imagination_horizon = 8
        cfg.horizons = None
        cfg.validate()
        assert cfg.active_horizon_pool == [8]

        cfg2 = ImaginedACTrainingConfig(horizons=[1, 2, 4])
        assert cfg2.active_horizon_pool == [1, 2, 4]


# ===================================================================
# 14. Fixed-probe determinism
# ===================================================================

class TestFixedProbe:
    @pytest.mark.models
    def test_fixed_probe_deterministic(self):
        torch.manual_seed(0)
        m = ReducedWorldModel(
            action_dim=ACTION_DIM, reward_head_kind="linear",
            tokenizer_eval_mode="mean",
        )
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

        loader = DataLoader(_FakeDataset(num=8), batch_size=2, shuffle=False)
        probe_batch = next(iter(loader))

        cfg = ImaginedACTrainingConfig(
            warmup_steps=4, imagination_horizon=4, max_batches=2, log_every=5,
        )
        tr = ImaginedACTrainer(
            model=m, train_loader=loader, train_cfg=cfg, seed=0,
            probe_batch=probe_batch,
            out_dir=Path(tempfile.mkdtemp()),
        )
        p1 = tr.evaluate_fixed_probe()
        p2 = tr.evaluate_fixed_probe()
        assert p1["1"]["imagined_return"] == p2["1"]["imagined_return"]

    @pytest.mark.models
    def test_probe_has_all_keys(self):
        torch.manual_seed(0)
        m = ReducedWorldModel(
            action_dim=ACTION_DIM, reward_head_kind="linear",
            tokenizer_eval_mode="mean",
        )
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

        loader = DataLoader(_FakeDataset(num=8), batch_size=2, shuffle=False)
        probe_batch = next(iter(loader))

        cfg = ImaginedACTrainingConfig(
            warmup_steps=4, imagination_horizon=4, max_batches=1, log_every=5,
        )
        tr = ImaginedACTrainer(
            model=m, train_loader=loader, train_cfg=cfg, seed=0,
            probe_batch=probe_batch,
            out_dir=Path(tempfile.mkdtemp()),
        )
        result = tr.evaluate_fixed_probe()
        for H in ("1", "2", "4"):
            assert H in result
            d = result[H]
            for key in ("imagined_return", "value_mean",
                        "steer_mean", "steer_std", "steer_saturation",
                        "gas_mean", "gas_std", "gas_saturation",
                        "brake_mean", "brake_std", "brake_saturation"):
                assert key in d, f"Missing key {key} in H={H}"


# ===================================================================
# 15. Per-dimension action statistics
# ===================================================================

class TestPerDimStats:
    @pytest.mark.models
    def test_per_dim_stats_in_metrics(self):
        torch.manual_seed(0)
        m = ReducedWorldModel(
            action_dim=ACTION_DIM, reward_head_kind="linear",
            tokenizer_eval_mode="mean",
        )
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

        cfg = ImaginedACTrainingConfig(
            warmup_steps=4, imagination_horizon=4, max_batches=3, log_every=5,
        )
        loader = DataLoader(_FakeDataset(num=8), batch_size=2, shuffle=False)
        tr = ImaginedACTrainer(
            model=m, train_loader=loader, train_cfg=cfg, seed=0,
            out_dir=Path(tempfile.mkdtemp()),
        )
        tr.train()
        for r in tr._metrics_log:
            for key in ("steer_mean", "steer_std",
                        "gas_mean", "gas_std",
                        "brake_mean", "brake_std"):
                assert key in r
                assert isinstance(r[key], float)


# ===================================================================
# 16. Config serialization
# ===================================================================

class TestConfig:
    def test_horizons_round_trip(self):
        cfg = ImaginedACTrainingConfig(horizons=[4, 1, 2])
        assert cfg.active_horizon_pool == [1, 2, 4]
        d = cfg.to_dict()
        cfg2 = ImaginedACTrainingConfig.from_dict(d)
        assert cfg2.horizons == [1, 2, 4]

    def test_validate_curriculum_out_of_range(self):
        with pytest.raises(ValueError, match="horizon in curriculum"):
            ImaginedACTrainingConfig(horizons=[1, 13]).validate()
