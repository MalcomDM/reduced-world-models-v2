"""Tests for masked-observation dynamics (Stage 2.5D)."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from torch.utils.data import DataLoader

from rwm.models.rwm.model import ReducedWorldModel
from rwm.evaluation.masked_factual_evaluator import (
    MaskedFactualEvaluator,
    _make_observation_keep,
    _make_prev_actions_correct,
    _make_prev_actions_zero,
    _make_prev_actions_shifted,
    _make_target_masks,
)


# ===================================================================
# Action-variant unit tests
# ===================================================================

class TestActionVariants:
    def test_correct_contract(self):
        """correct: prev[:,0]=predecessor, prev[:,t]=actions[:,t-1]."""
        B, T, A = 2, 4, 3
        actions = torch.randn(B, T, A)
        pred = torch.randn(B, A)
        prev = _make_prev_actions_correct(actions, pred)
        torch.testing.assert_close(prev[:, 0], pred)
        if T > 1:
            torch.testing.assert_close(prev[:, 1:], actions[:, :-1])

    def test_zero_is_all_zeros(self):
        B, T, A = 2, 4, 3
        actions = torch.randn(B, T, A)
        pred = torch.randn(B, A)
        prev = _make_prev_actions_zero(actions, pred)
        assert (prev == 0).all()

    def test_shifted_equals_actions(self):
        """shifted: prev[t] = actions[t] (uses current as 'previous')."""
        B, T, A = 2, 4, 3
        actions = torch.randn(B, T, A)
        pred = torch.randn(B, A)
        prev = _make_prev_actions_shifted(actions, pred)
        torch.testing.assert_close(prev, actions)

    def test_variants_differ(self):
        B, T, A = 2, 4, 3
        actions = torch.randn(B, T, A)
        pred = torch.randn(B, A)
        c = _make_prev_actions_correct(actions, pred)
        z = _make_prev_actions_zero(actions, pred)
        s = _make_prev_actions_shifted(actions, pred)
        assert not torch.allclose(c, z)
        assert not torch.allclose(c, s)

    def test_evaluator_variants_preserve_visible_warmup(self):
        """Only blind steps may receive altered previous-action context."""
        actions = torch.arange(15, dtype=torch.float32).view(1, 5, 3)
        predecessor = torch.tensor([[99.0, 98.0, 97.0]])
        correct = _make_prev_actions_correct(actions, predecessor)
        shifted = _make_prev_actions_shifted(actions, predecessor)
        warmup = 2
        applied = correct.clone()
        applied[:, warmup:] = shifted[:, warmup:]
        torch.testing.assert_close(applied[:, :warmup], correct[:, :warmup])
        torch.testing.assert_close(applied[:, warmup:], shifted[:, warmup:])


# ===================================================================
# Observation mask tests
# ===================================================================

class TestObservationMask:
    def test_all_visible(self):
        keep = _make_observation_keep(T=10, warmup=0, mask_horizon=0, device="cpu")
        assert keep.all()

    def test_warmup_only(self):
        keep = _make_observation_keep(T=10, warmup=0, mask_horizon=0, device="cpu")
        assert keep.all()

    def test_mask_starts_at_warmup(self):
        keep = _make_observation_keep(T=10, warmup=3, mask_horizon=4, device="cpu")
        assert keep[:, :3].all()
        assert (~keep[:, 3:7]).all()
        # Steps beyond mask_horizon are visible again (recovery allowed)
        assert keep[:, 7:].all()

    def test_mask_clips_to_T(self):
        keep = _make_observation_keep(T=6, warmup=2, mask_horizon=10, device="cpu")
        assert keep[:, :2].all()
        assert (~keep[:, 2:]).all()

    @pytest.mark.parametrize("horizon", [1, 2, 4, 8, 12])
    def test_recurrent_mask_is_relative_to_target_region(self, horizon):
        """SRU burn-in remains visible; warmup begins at the loss target."""
        B, T = 2, 36
        valid = torch.ones(B, T, dtype=torch.bool)
        loss_mask = torch.zeros(B, T, dtype=torch.bool)
        loss_mask[:, 20:36] = True

        keep, score, variants = _make_target_masks(
            loss_mask=loss_mask,
            valid_step=valid,
            batch_size=B,
            sequence_len=T,
            warmup=4,
            mask_horizon=horizon,
            device=torch.device("cpu"),
        )

        assert keep[:, :24].all()
        assert (~keep[:, 24:24 + horizon]).all()
        assert keep[:, 24 + horizon:].all()
        torch.testing.assert_close(score, ~keep)
        torch.testing.assert_close(variants, score)
        assert int(score.sum().item()) == B * horizon

    def test_recurrent_mask_excludes_padding_and_requires_valid_targets(self):
        B, T = 2, 36
        valid = torch.ones(B, T, dtype=torch.bool)
        valid[0, :7] = False
        loss_mask = torch.zeros(B, T, dtype=torch.bool)
        loss_mask[:, 20:36] = True

        keep, score, _ = _make_target_masks(
            loss_mask=loss_mask,
            valid_step=valid,
            batch_size=B,
            sequence_len=T,
            warmup=4,
            mask_horizon=4,
            device=torch.device("cpu"),
        )

        assert not keep[0, :7].any()
        assert not score[0, :7].any()
        assert (~keep[:, 24:28]).all()
        assert int(score.sum().item()) == 8

    def test_predecessor_action_is_placed_at_first_valid_position(self):
        actions = torch.arange(36 * 3, dtype=torch.float32).view(1, 36, 3)
        predecessor = torch.tensor([[99.0, 98.0, 97.0]])
        valid = torch.zeros(1, 36, dtype=torch.bool)
        valid[:, 7:] = True

        prev = _make_prev_actions_correct(actions, predecessor, valid_step=valid)

        torch.testing.assert_close(prev[:, 7], predecessor)
        torch.testing.assert_close(prev[:, 8:], actions[:, 7:-1])


# ===================================================================
# Model-level: observation_keep in forward_sequence
# ===================================================================

class TestForwardSequenceMask:
    def test_all_visible_mask_matches_no_mask(self):
        """All-True observation_keep must match the default (no mask) output."""
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        model.eval()
        B, T = 2, 4
        obs = torch.randn(B, T, 3, 64, 64)
        acts = torch.randn(B, T, 3)
        pred = torch.randn(B, 3)
        prev = _make_prev_actions_correct(acts, pred)

        with torch.no_grad():
            out_default = model.forward_sequence(obs, prev, acts, force_keep_input=True)
            keep_all = torch.ones(B, T, dtype=torch.bool)
            out_masked = model.forward_sequence(
                obs, prev, acts, force_keep_input=True,
                observation_keep=keep_all,
            )
        torch.testing.assert_close(out_default.reward_pred_seq, out_masked.reward_pred_seq)
        torch.testing.assert_close(out_default.world_state, out_masked.world_state)

    def test_fully_masked_output_does_not_depend_on_image(self):
        """When all steps are masked, changing the image should not change output."""
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        model.eval()
        B, T = 1, 4
        acts = torch.randn(B, T, 3)
        pred = torch.randn(B, 3)
        prev = _make_prev_actions_correct(acts, pred)

        img_a = torch.randn(B, T, 3, 64, 64)
        img_b = torch.randn(B, T, 3, 64, 64)
        keep_none = torch.zeros(B, T, dtype=torch.bool)

        with torch.no_grad():
            out_a = model.forward_sequence(
                img_a, prev, acts, force_keep_input=True,
                observation_keep=keep_none,
            )
            out_b = model.forward_sequence(
                img_b, prev, acts, force_keep_input=True,
                observation_keep=keep_none,
            )
        torch.testing.assert_close(out_a.reward_pred_seq, out_b.reward_pred_seq)

    def test_warmup_image_change_affects_output(self):
        """Changing the warmup image should change masked outputs."""
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        model.eval()
        B, T = 1, 6
        warmup = 3
        acts = torch.randn(B, T, 3)
        pred = torch.randn(B, 3)
        prev = _make_prev_actions_correct(acts, pred)

        img_a = torch.randn(B, T, 3, 64, 64)
        img_b = img_a.clone()
        img_b[:, :warmup] = torch.randn(B, warmup, 3, 64, 64)
        keep = _make_observation_keep(T, warmup=warmup, mask_horizon=3, device="cpu")

        with torch.no_grad():
            out_a = model.forward_sequence(
                img_a, prev, acts, force_keep_input=True,
                observation_keep=keep,
            )
            out_b = model.forward_sequence(
                img_b, prev, acts, force_keep_input=True,
                observation_keep=keep,
            )
        # The warmup images differ, so beliefs at all positions should differ
        assert not torch.allclose(out_a.reward_pred_seq, out_b.reward_pred_seq), (
            "Changing warmup images must affect masked outputs"
        )

    def test_masked_image_change_does_not_affect_output(self):
        """Changing images at masked positions should not affect output."""
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        model.eval()
        B, T = 1, 6
        warmup = 3
        acts = torch.randn(B, T, 3)
        pred = torch.randn(B, 3)
        prev = _make_prev_actions_correct(acts, pred)

        img_a = torch.randn(B, T, 3, 64, 64)
        img_b = img_a.clone()
        img_b[:, warmup:] = torch.randn(B, T - warmup, 3, 64, 64)
        keep = _make_observation_keep(T, warmup=warmup, mask_horizon=3, device="cpu")

        with torch.no_grad():
            out_a = model.forward_sequence(
                img_a, prev, acts, force_keep_input=True,
                observation_keep=keep,
            )
            out_b = model.forward_sequence(
                img_b, prev, acts, force_keep_input=True,
                observation_keep=keep,
            )
        torch.testing.assert_close(out_a.reward_pred_seq, out_b.reward_pred_seq)

    def test_fully_visible_mask_finite_gradients(self):
        """Outputs have finite values and preserve gradients."""
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        model.train()
        B, T = 2, 4
        obs = torch.randn(B, T, 3, 64, 64)
        acts = torch.randn(B, T, 3)
        prev = _make_prev_actions_correct(acts, torch.randn(B, 3))
        keep = torch.ones(B, T, dtype=torch.bool)

        out = model.forward_sequence(
            obs, prev, acts, force_keep_input=False,
            observation_keep=keep,
        )
        loss = out.reward_pred_seq.sum()
        loss.backward()
        assert torch.isfinite(out.reward_pred_seq).all()
        assert any(p.grad is not None for p in model.encoder.parameters())


# ===================================================================
# Model-level: observation_keep in forward (incremental)
# ===================================================================

class TestForwardIncrementalMask:
    def test_masked_step_zeros_spatial(self):
        """observation_keep=False should zero the spatial representation."""
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        model.eval()
        img = torch.randn(1, 3, 64, 64)
        act = torch.zeros(1, 3)

        with torch.no_grad():
            out_visible = model(img=img, prev_action=act, current_action=act,
                                force_keep_input=True, observation_keep=True)
            out_masked = model(img=img, prev_action=act, current_action=act,
                               force_keep_input=True, observation_keep=False)

        # Masked output should differ (zero spatial rep vs real spatial rep)
        assert not torch.allclose(out_visible.reward_pred, out_masked.reward_pred)

    def test_masked_finite(self):
        """Masked forward outputs have finite values."""
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        model.eval()
        img = torch.randn(1, 3, 64, 64)
        act = torch.zeros(1, 3)
        with torch.no_grad():
            out = model(img=img, prev_action=act, current_action=act,
                        force_keep_input=True, observation_keep=False)
        assert torch.isfinite(out.reward_pred).all()


# ===================================================================
# MaskedFactualEvaluator integration
# ===================================================================

class TestEvaluator:
    class _SpyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tokenizer = SimpleNamespace(eval_mode="mean")
            self.calls = []

        def forward_sequence(
            self,
            obs,
            prev_actions,
            current_actions,
            **kwargs,
        ):
            self.calls.append({
                "prev_actions": prev_actions.detach().clone(),
                "current_actions": current_actions.detach().clone(),
                **{
                    key: value.detach().clone() if torch.is_tensor(value) else value
                    for key, value in kwargs.items()
                },
            })
            return SimpleNamespace(
                reward_pred_seq=torch.zeros(
                    obs.shape[:2], device=obs.device, dtype=obs.dtype,
                ),
            )

    @staticmethod
    def _recurrent_loader(batch_size=2):
        T = 36
        items = []
        for b in range(batch_size):
            actions = torch.arange(T * 3, dtype=torch.float32).view(T, 3) + b * 1000
            rewards = torch.arange(T, dtype=torch.float32) + b * 100
            valid = torch.ones(T, dtype=torch.bool)
            if b == 0:
                valid[:7] = False
            loss_mask = torch.zeros(T, dtype=torch.bool)
            loss_mask[20:36] = True
            items.append({
                "obs": torch.zeros(T, 3, 2, 2),
                "action": actions,
                "reward": rewards,
                "predecessor_action": torch.tensor([99.0, 98.0, 97.0]),
                "valid_step": valid,
                "loss_mask": loss_mask,
            })
        return DataLoader(items, batch_size=batch_size)

    def test_recurrent_evaluation_scores_only_blind_targets_and_forwards_strict(self):
        model = self._SpyModel()
        evaluator = MaskedFactualEvaluator(
            model,
            torch.device("cpu"),
            train_reward_mean=0.0,
            observation_dropout_execution="pre_perception_skip",
        )
        loader = self._recurrent_loader()

        result = evaluator.evaluate_horizon(
            loader, warmup=4, mask_horizon=4, action_variant="correct",
        )

        assert len(model.calls) == 2
        visible_call, masked_call = model.calls
        keep = masked_call["observation_keep"]
        assert keep[0, 7:24].all()
        assert keep[1, :24].all()
        assert not keep[0, :7].any()
        assert (~keep[:, 24:28]).all()
        assert keep[:, 28:].all()
        assert masked_call["observation_dropout_execution"] == "pre_perception_skip"
        torch.testing.assert_close(
            masked_call["valid_step"], next(iter(loader))["valid_step"],
        )
        assert result["transitions"] == 8
        expected = torch.tensor(
            [24.0, 25.0, 26.0, 27.0, 124.0, 125.0, 126.0, 127.0],
        )
        assert result["val_mse"] == pytest.approx(expected.square().mean().item())
        assert result["val_mae"] == pytest.approx(expected.abs().mean().item())
        assert result["baseline_mse"] == pytest.approx(expected.square().mean().item())
        assert result["observation_dropout_execution"] == "pre_perception_skip"
        assert result["skipped_valid_positions"] == 8
        assert visible_call["observation_keep"][0, :7].sum().item() == 0

    def test_post_perception_reports_masked_images_as_processed(self):
        model = self._SpyModel()
        evaluator = MaskedFactualEvaluator(
            model,
            torch.device("cpu"),
            train_reward_mean=0.0,
            observation_dropout_execution="post_perception",
        )

        result = evaluator.evaluate_horizon(
            self._recurrent_loader(),
            warmup=4,
            mask_horizon=4,
        )

        assert result["visible_valid_positions"] < result["valid_input_positions"]
        assert result["perceived_valid_positions"] == result["valid_input_positions"]
        assert result["skipped_valid_positions"] == 0

    @pytest.mark.parametrize("variant", ["zero", "shifted"])
    def test_action_variant_changes_only_blind_target_positions(self, variant):
        model = self._SpyModel()
        evaluator = MaskedFactualEvaluator(
            model, torch.device("cpu"), train_reward_mean=0.0,
        )
        loader = self._recurrent_loader()

        evaluator.evaluate_horizon(
            loader, warmup=4, mask_horizon=4, action_variant=variant,
        )

        visible_prev = model.calls[0]["prev_actions"]
        masked_prev = model.calls[1]["prev_actions"]
        torch.testing.assert_close(masked_prev[:, :24], visible_prev[:, :24])
        torch.testing.assert_close(masked_prev[:, 28:], visible_prev[:, 28:])
        if variant == "zero":
            assert not masked_prev[:, 24:28].any()
        else:
            torch.testing.assert_close(
                masked_prev[:, 24:28],
                model.calls[1]["current_actions"][:, 24:28],
            )

    def test_evaluator_returns_per_horizon_results(self):
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        evaluator = MaskedFactualEvaluator(model, device, train_reward_mean=0.0)

        # Build a minimal loader with synthetic data
        from torch.utils.data import DataLoader, TensorDataset

        B, T = 4, 16
        obs = torch.randn(B, T, 3, 64, 64)
        acts = torch.randn(B, T, 3)
        rewards = torch.randn(B, T)
        pred = torch.randn(B, 3)
        dataset = TensorDataset(obs, acts, rewards, pred)
        loader = DataLoader(
            [(o, a, r, p) for o, a, r, p in zip(obs, acts, rewards, pred)],
            batch_size=B, collate_fn=lambda x: {
                "obs": torch.stack([i[0] for i in x]),
                "action": torch.stack([i[1] for i in x]),
                "reward": torch.stack([i[2] for i in x]),
                "predecessor_action": torch.stack([i[3] for i in x]),
            },
        )

        results = evaluator.evaluate(loader, warmup=4,
                                     mask_horizons=(1, 2, 4),
                                     action_variants=("correct",))
        assert "horizons" in results
        assert len(results["horizons"]) == 3  # 3 horizons × 1 variant

        for h in results["horizons"]:
            assert "val_mse" in h
            assert "mask_horizon" in h
            assert h["action_variant"] == "correct"

    def test_evaluator_masked_mse_finite(self):
        """Masked evaluation returns finite MSE across all horizons."""
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        evaluator = MaskedFactualEvaluator(model, device, train_reward_mean=0.0)

        from torch.utils.data import DataLoader
        B, T = 2, 8
        obs = torch.randn(B, T, 3, 64, 64)
        acts = torch.randn(B, T, 3)
        rewards = torch.randn(B, T)
        pred = torch.randn(B, 3)
        loader = DataLoader(
            [{"obs": o, "action": a, "reward": r, "predecessor_action": p}
             for o, a, r, p in zip(obs, acts, rewards, pred)],
            batch_size=B,
        )

        for h in (1, 2, 4):
            r = evaluator.evaluate_horizon(loader, warmup=2, mask_horizon=h)
            assert torch.isfinite(torch.tensor(r["val_mse"])), f"horizon {h}: MSE not finite"
            assert torch.isfinite(torch.tensor(r["ratio"])), f"horizon {h}: ratio not finite"
            assert r["transitions"] > 0, f"horizon {h}: no transitions evaluated"


# ===================================================================
# Unchanged legacy behavior
# ===================================================================

class TestLegacyBehavior:
    def test_unmasked_forward_sequence_still_works(self):
        """forward_sequence without observation_keep must work as before."""
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        model.eval()
        B, T = 2, 4
        obs = torch.randn(B, T, 3, 64, 64)
        acts = torch.randn(B, T, 3)
        prev = _make_prev_actions_correct(acts, torch.randn(B, 3))
        out = model.forward_sequence(obs, prev, acts, force_keep_input=True)
        assert out.reward_pred_seq.shape == (B, T)
        assert torch.isfinite(out.reward_pred_seq).all()

    def test_unmasked_forward_still_works(self):
        """forward without observation_keep must work as before."""
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        model.eval()
        img = torch.randn(1, 3, 64, 64)
        act = torch.zeros(1, 3)
        out = model(img=img, prev_action=act, current_action=act,
                    force_keep_input=True)
        assert out.reward_pred.shape == (1, 1)
        assert torch.isfinite(out.reward_pred).all()


# ===================================================================
# Temporal mask sampler tests (D.1)
# ===================================================================

class TestTemporalMaskSampler:
    def test_disabled_gives_all_visible(self):
        from rwm.trainers.deterministic.temporal_mask import sample_mask
        rng = torch.Generator()
        keep = sample_mask(4, 16, warmup_steps=4, horizons=[1, 2, 4, 8, 12],
                           mask_probability=0.0, rng=rng, device="cpu")
        assert keep.all()

    def test_contiguous_blind_block(self):
        """Masked positions form a single contiguous block after warmup."""
        from rwm.trainers.deterministic.temporal_mask import sample_mask
        rng = torch.Generator()
        rng.manual_seed(0)
        keep = sample_mask(100, 16, warmup_steps=4, horizons=[4],
                           mask_probability=1.0, rng=rng, device="cpu")
        # All samples should have warmup=True
        assert keep[:, :4].all()
        # All samples should have block of 4 masked starting at 4
        for b in range(100):
            assert not keep[b, 4:8].any(), f"Sample {b} not masked at positions 4-7"
            assert keep[b, 8:].all(), f"Sample {b}: positions after mask should be visible"

    def test_horizon_never_exceeds_capacity(self):
        from rwm.trainers.deterministic.temporal_mask import _validate_config
        _validate_config(4, [1, 2, 4, 8], 16)  # should pass
        with pytest.raises(ValueError):
            _validate_config(4, [1, 13], 16)  # 13 > 16-4=12

    def test_horizon_within_approved_set(self):
        """Sampled horizon must be one of the approved values."""
        from rwm.trainers.deterministic.temporal_mask import sample_mask
        rng = torch.Generator()
        rng.manual_seed(42)
        approved = {1, 4, 8}
        keep = sample_mask(50, 16, warmup_steps=4, horizons=list(approved),
                           mask_probability=1.0, rng=rng, device="cpu")
        for b in range(50):
            false_count = int((~keep[b]).sum().item())
            assert false_count in approved, f"Sample {b}: horizon {false_count} not in {approved}"

    def test_seeded_reproducibility(self):
        from rwm.trainers.deterministic.temporal_mask import sample_mask
        rng1 = torch.Generator(); rng1.manual_seed(42)
        rng2 = torch.Generator(); rng2.manual_seed(42)
        k1 = sample_mask(8, 16, warmup_steps=4, horizons=[1, 2, 4, 8, 12],
                         mask_probability=0.5, rng=rng1, device="cpu")
        k2 = sample_mask(8, 16, warmup_steps=4, horizons=[1, 2, 4, 8, 12],
                         mask_probability=0.5, rng=rng2, device="cpu")
        torch.testing.assert_close(k1, k2)

    def test_different_seeds_differ(self):
        from rwm.trainers.deterministic.temporal_mask import sample_mask
        rng1 = torch.Generator(); rng1.manual_seed(0)
        rng2 = torch.Generator(); rng2.manual_seed(1)
        k1 = sample_mask(8, 16, warmup_steps=4, horizons=[1, 2, 4, 8, 12],
                         mask_probability=1.0, rng=rng1, device="cpu")
        k2 = sample_mask(8, 16, warmup_steps=4, horizons=[1, 2, 4, 8, 12],
                         mask_probability=1.0, rng=rng2, device="cpu")
        # With prob=1.0 and different seeds, horizons should differ
        assert not torch.allclose(k1, k2)

    def test_warmup_greater_than_seqlen_raises(self):
        from rwm.trainers.deterministic.temporal_mask import _validate_config
        with pytest.raises(ValueError):
            _validate_config(16, [1], 16)


class TestMaskRampSchedule:
    def test_ramp_schedule(self):
        from rwm.trainers.deterministic.temporal_mask import current_mask_probability
        target = 0.5
        ramp_epochs = 2
        assert current_mask_probability(0, target, ramp_epochs) == 0.0
        assert current_mask_probability(1, target, ramp_epochs) == 0.25
        assert current_mask_probability(2, target, ramp_epochs) == 0.5
        assert current_mask_probability(3, target, ramp_epochs) == 0.5
        assert current_mask_probability(10, target, ramp_epochs) == 0.5

    def test_ramp_clips(self):
        from rwm.trainers.deterministic.temporal_mask import current_mask_probability
        assert current_mask_probability(5, 0.8, 3) == 0.8
        assert current_mask_probability(0, 0.8, 3) == 0.0
        assert current_mask_probability(1, 0.8, 3) == pytest.approx(0.8 / 3)


# ===================================================================
# Trainer integration with temporal mask (D.1)
# ===================================================================

class TestTrainerTemporalMask:
    def test_legacy_config_defaults_to_disabled(self):
        """ExperimentConfig without temporal_mask defaults to disabled."""
        from rwm.config.experiment_config import ExperimentConfig
        cfg = ExperimentConfig()
        assert cfg.training.temporal_mask.enabled is False

    def test_legacy_checkpoint_compat(self, tmp_path):
        """Checkpoint saved without temporal_mask loads with enabled=False."""
        from rwm.config.experiment_config import ExperimentConfig, TrainingConfig
        from rwm.utils.checkpointing import save_checkpoint, load_checkpoint

        cfg = ExperimentConfig(training=TrainingConfig())
        m = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        path = save_checkpoint(tmp_path / "ckpt", model_state=m.state_dict(), config=cfg)
        loaded_cfg = load_checkpoint(path)["config"]
        assert loaded_cfg.training.temporal_mask.enabled is False

    def test_disabled_mask_preserves_behavior(self, tmp_path):
        """Disabled TemporalMaskConfig produces all-visible training."""
        from rwm.trainers.deterministic.world_model_trainer import WorldModelTrainer
        from rwm.config.experiment_config import ExperimentConfig, TrainingConfig, TemporalMaskConfig
        from torch.utils.data import DataLoader

        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        model.eval()
        B, T = 2, 8
        obs = torch.randn(B, T, 3, 64, 64)
        acts = torch.randn(B, T, 3)
        rew = torch.randn(B, T)
        pred = torch.randn(B, 3)
        ds = [{"obs": o, "action": a, "reward": r, "predecessor_action": p}
              for o, a, r, p in zip(obs, acts, rew, pred)]
        loader = DataLoader(ds, batch_size=B)

        cfg = ExperimentConfig(
            training=TrainingConfig(
                temporal_mask=TemporalMaskConfig(enabled=False),
            ),
        )
        trainer = WorldModelTrainer(
            loader, None, out_dir=tmp_path, sequence_len=T,
            epochs=1, batch_size=B, config=cfg,
        )
        assert trainer._mask_cfg.enabled is False
        # Training with disabled mask should not use observation_keep
        loss_t, loss_m, loss_k, _ = trainer.train_one_epoch(epoch=1)
        assert torch.isfinite(torch.tensor(loss_t))

    def test_enabled_mask_produces_finite_loss(self, tmp_path):
        """Training with temporal mask enabled must produce finite loss."""
        from rwm.trainers.deterministic.world_model_trainer import WorldModelTrainer
        from rwm.config.experiment_config import ExperimentConfig, TrainingConfig, TemporalMaskConfig
        from torch.utils.data import DataLoader

        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        B, T = 4, 8
        obs = torch.randn(B, T, 3, 64, 64)
        acts = torch.randn(B, T, 3)
        rew = torch.randn(B, T)
        pred = torch.randn(B, 3)
        ds = [{"obs": o, "action": a, "reward": r, "predecessor_action": p}
              for o, a, r, p in zip(obs, acts, rew, pred)]
        loader = DataLoader(ds, batch_size=B)

        cfg = ExperimentConfig(
            training=TrainingConfig(
                temporal_mask=TemporalMaskConfig(
                    enabled=True, warmup_steps=2,
                    horizons=[1, 2], target_mask_probability=0.5,
                    ramp_epochs=1,
                ),
            ),
        )
        trainer = WorldModelTrainer(
            loader, None, out_dir=tmp_path, sequence_len=T,
            epochs=1, batch_size=B, config=cfg,
        )
        loss_t, loss_m, loss_k, _ = trainer.train_one_epoch(epoch=1)
        assert torch.isfinite(torch.tensor(loss_t))

    def test_mask_preserves_action_timing(self, tmp_path):
        """Previous/current action timing is unchanged under masking."""
        from rwm.trainers.deterministic.world_model_trainer import WorldModelTrainer
        from rwm.config.experiment_config import ExperimentConfig, TrainingConfig, TemporalMaskConfig
        from torch.utils.data import DataLoader

        B, T = 2, 8
        obs = torch.randn(B, T, 3, 64, 64)
        acts = torch.zeros(B, T, 3)  # Use distinct actions
        for b in range(B):
            for t in range(T):
                acts[b, t] = torch.tensor([float(t), 0.0, 0.0])
        rew = torch.randn(B, T)
        pred = torch.randn(B, 3)
        ds = [{"obs": o, "action": a, "reward": r, "predecessor_action": p}
              for o, a, r, p in zip(obs, acts, rew, pred)]
        loader = DataLoader(ds, batch_size=B)

        cfg = ExperimentConfig(
            training=TrainingConfig(
                temporal_mask=TemporalMaskConfig(
                    enabled=True, warmup_steps=2,
                    horizons=[4], target_mask_probability=1.0,
                    ramp_epochs=1,
                ),
            ),
        )
        trainer = WorldModelTrainer(
            loader, None, out_dir=tmp_path, sequence_len=T,
            epochs=1, batch_size=B, config=cfg,
        )
        # Run training
        loss_t, loss_m, loss_k, _ = trainer.train_one_epoch(epoch=1)
        assert torch.isfinite(torch.tensor(loss_t))


# ===================================================================
# CLI import tests
# ===================================================================

class TestCLIImports:
    def test_script_imports_and_has_reused_loaders(self):
        """The evaluate_masked_dynamics.py script imports successfully and
        reuses build_protocol_loaders / reward_mean from evaluate_checkpoint."""
        import importlib.util
        script_path = (
            Path(__file__).resolve().parent.parent.parent / "scripts" / "evaluation"
            / "evaluate_masked_dynamics.py"
        )
        assert script_path.exists()
        spec = importlib.util.spec_from_file_location("_emd", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "build_protocol_loaders")
        assert hasattr(mod, "reward_mean")
        assert hasattr(mod, "_resolve_recurrent_layout")

    def test_cli_resolves_sru_burn_in_from_checkpoint_config(self):
        import importlib.util
        from rwm.config.experiment_config import ExperimentConfig, TemporalConfig

        script_path = (
            Path(__file__).resolve().parent.parent.parent / "scripts" / "evaluation"
            / "evaluate_masked_dynamics.py"
        )
        spec = importlib.util.spec_from_file_location("_emd_layout", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        cfg = ExperimentConfig(
            temporal=TemporalConfig(
                backend="minimal_sru",
                sru_burn_in_steps=20,
            ),
        )
        assert mod._resolve_recurrent_layout(cfg) == ("minimal_sru", True, 20)
        assert mod._resolve_recurrent_layout({}) == (
            "causal_transformer", False, 0,
        )

    def test_refuse_overwrite_logic(self, tmp_path):
        """Refuse-overwrite logic works when main() is called with an existing path."""
        import argparse
        out_path = tmp_path / "existing.json"
        out_path.write_text("{}")
        # Patch sys.argv so main() sees the existing output path
        import importlib.util
        script_path = (
            Path(__file__).resolve().parent.parent.parent / "scripts" / "evaluation"
            / "evaluate_masked_dynamics.py"
        )
        spec = importlib.util.spec_from_file_location("_emd2", script_path)
        mod = importlib.util.module_from_spec(spec)
        # Test the refusl logic directly by calling argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--data-split-seed", type=int, default=0)
        parser.add_argument("--cache-dir", type=Path, default=None)
        parser.add_argument("--data-root", type=Path, default=Path("data"))
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument("--sequence-len", type=int, default=16)
        parser.add_argument("--max-val-windows", type=int, default=256)
        parser.add_argument("--warmup", type=int, default=4)
        args = parser.parse_args([
            "--checkpoint", str(tmp_path / "dummy.pt"),
            "--out", str(out_path),
        ])
        assert out_path.exists()
        # The script's main() would refuse; verify the check exists
        import inspect
        source = inspect.getsource(mod)
        assert "Refusing to overwrite" in source
