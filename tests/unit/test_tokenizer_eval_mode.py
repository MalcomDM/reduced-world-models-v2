"""Tests for tokenizer evaluation-mode configuration (C.3)."""

import copy
from pathlib import Path

import pytest
import torch

from rwm.models.rwm.model import ReducedWorldModel
from rwm.models.rwm.tokenization_head import TokenizationHead
from rwm.config.experiment_config import ExperimentConfig, PerceptionConfig
from rwm.utils.checkpointing import save_checkpoint, load_checkpoint, model_from_checkpoint


# ===================================================================
# Unit: TokenizationHead eval_mode
# ===================================================================

class TestEvalModeConstruction:
    def test_default_is_sample(self):
        head = TokenizationHead()
        assert head.eval_mode == "sample"

    def test_mean_roundtrip(self):
        head = TokenizationHead(eval_mode="mean")
        assert head.eval_mode == "mean"
        head.eval()
        from rwm.models.rwm.encoder import Encoder
        encoder = Encoder().eval()
        img = torch.randn(2, 3, 64, 64)
        feat = encoder(img)
        with torch.no_grad():
            a, _, _ = head(feat)
            b, _, _ = head(feat)
        torch.testing.assert_close(a, b)

    def test_invalid_mode_raises(self):
        with pytest.raises(AssertionError):
            TokenizationHead(eval_mode="invalid")


# ===================================================================
# Model-level: eval mode threading
# ===================================================================

class TestModelEvalMode:
    def test_default_is_sample(self):
        model = ReducedWorldModel(action_dim=3)
        assert model._tokenizer_eval_mode == "sample"
        assert model.tokenizer.eval_mode == "sample"

    def test_constructor_threads_mean(self):
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        assert model.tokenizer.eval_mode == "mean"

    def test_mean_eval_deterministic(self):
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        model.eval()
        img = torch.randn(1, 3, 64, 64)
        act = torch.zeros(1, 3)
        with torch.no_grad():
            o1 = model(img=img, prev_action=act, current_action=act, force_keep_input=True)
            o2 = model(img=img, prev_action=act, current_action=act, force_keep_input=True)
        torch.testing.assert_close(o1.reward_pred, o2.reward_pred)

    def test_sample_eval_can_vary(self):
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="sample")
        model.eval()
        img = torch.randn(1, 3, 64, 64)
        act = torch.zeros(1, 3)
        # Different seeds -> likely different
        predictions = []
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            with torch.no_grad():
                o = model(img=img, prev_action=act, current_action=act, force_keep_input=True)
            predictions.append(o.reward_pred.item())
        unique = len(set(round(p, 6) for p in predictions))
        # Not strictly guaranteed but extremely likely with 3 seeds
        assert unique > 1, (
            f"Expected varying predictions with stemd seeds, got all {predictions[0]}"
        )

    def test_training_stochastic_even_with_mean_mode(self):
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        model.train()
        img = torch.randn(1, 3, 64, 64)
        act = torch.zeros(1, 3)
        with torch.no_grad():
            o1 = model(img=img, prev_action=act, current_action=act, force_keep_input=True)
            o2 = model(img=img, prev_action=act, current_action=act, force_keep_input=True)
        # Training is stochastic, outputs should differ
        assert not torch.allclose(o1.reward_pred, o2.reward_pred), (
            "Training must be stochastic even with eval_mode='mean'"
        )

    def test_forward_sequence_training_stochastic_with_mean_mode(self):
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        model.train()
        B, T = 2, 4
        obs = torch.randn(B, T, 3, 64, 64)
        prev_act = torch.zeros(B, T, 3)
        curr_act = torch.zeros(B, T, 3)
        with torch.no_grad():
            o1 = model.forward_sequence(obs, prev_act, curr_act)
            o2 = model.forward_sequence(obs, prev_act, curr_act)
        assert not torch.allclose(o1.reward_pred, o2.reward_pred)


# ===================================================================
# Checkpoint round-trip
# ===================================================================

class TestCheckpointRoundtrip:
    def test_structured_restores_mean(self, tmp_path):
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        cfg = ExperimentConfig(
            perception=PerceptionConfig(tokenizer_eval_mode="mean"),
        )
        path = save_checkpoint(tmp_path / "ckpt", model_state=model.state_dict(), config=cfg)
        ckpt = load_checkpoint(path)
        loaded = model_from_checkpoint(ckpt, action_dim=3)
        assert loaded._tokenizer_eval_mode == "mean"
        assert loaded.tokenizer.eval_mode == "mean"

    def test_structured_restores_sample(self, tmp_path):
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="sample")
        cfg = ExperimentConfig(
            perception=PerceptionConfig(tokenizer_eval_mode="sample"),
        )
        path = save_checkpoint(tmp_path / "ckpt", model_state=model.state_dict(), config=cfg)
        ckpt = load_checkpoint(path)
        loaded = model_from_checkpoint(ckpt, action_dim=3)
        assert loaded._tokenizer_eval_mode == "sample"
        assert loaded.tokenizer.eval_mode == "sample"

    def test_legacy_checkpoint_defaults_to_sample(self, tmp_path):
        """Legacy bare state_dict without perception config must restore as
        tokenizer_eval_mode='sample'."""
        model = ReducedWorldModel(action_dim=3)
        torch.save(model.state_dict(), tmp_path / "legacy.pt")
        ckpt = load_checkpoint(tmp_path / "legacy.pt")
        loaded = model_from_checkpoint(ckpt, action_dim=3)
        assert loaded._tokenizer_eval_mode == "sample"

    def test_structured_checkpoint_without_field_defaults_to_sample(self, tmp_path):
        """Structured checkpoint missing tokenizer_eval_mode in config loads as 'sample'."""
        model = ReducedWorldModel(action_dim=3)
        cfg = ExperimentConfig(
            perception=PerceptionConfig(selection_mode="learned", k=8),
        )
        path = save_checkpoint(tmp_path / "ckpt", model_state=model.state_dict(), config=cfg)
        ckpt = load_checkpoint(path)
        # Manually remove the field from config dict to simulate old checkpoint
        ckpt["config"] = ExperimentConfig.from_dict({
            "seed": 42,
            "perception": {"k": 8, "selection_mode": "learned", "selection_seed": 0},
        })
        loaded = model_from_checkpoint(ckpt, action_dim=3)
        assert loaded._tokenizer_eval_mode == "sample"


# ===================================================================
# Existing Stage-02 K=8 anchor still loads
# ===================================================================

class TestLegacyAnchorCompat:
    @pytest.fixture
    def stage02_path(self):
        p = Path("runs/component_refinement/02_vectorized_reward_anchor/beta0.1_seed42/checkpoint_best.pt")
        if not p.exists():
            pytest.skip("Stage-02 anchor checkpoint not available")
        return p

    def test_stage02_loads_as_sample(self, stage02_path):
        ckpt = load_checkpoint(stage02_path)
        loaded = model_from_checkpoint(ckpt, action_dim=3)
        assert loaded._tokenizer_eval_mode == "sample"
        assert loaded.tokenizer.eval_mode == "sample"

    def test_stage02_action_probe_runs(self, stage02_path):
        from rwm.utils.probe_set import make_default_probe
        ckpt = load_checkpoint(stage02_path)
        loaded = model_from_checkpoint(ckpt, action_dim=3)
        loaded.eval()
        probe_obs, probe_act = make_default_probe()
        obs_t = torch.tensor(probe_obs, dtype=torch.float32).permute(0, 3, 1, 2)
        act_t = torch.tensor(probe_act, dtype=torch.float32)
        with torch.no_grad():
            out = loaded(img=obs_t, prev_action=act_t, current_action=act_t, force_keep_input=True)
        assert out.reward_pred.shape == (8, 1)

    def test_k8_attention_trace_runs(self, stage02_path):
        from rwm.evaluation.attention_trace import trace_attention
        ckpt = load_checkpoint(stage02_path)
        loaded = model_from_checkpoint(ckpt, action_dim=3)
        loaded.eval()
        img = torch.randn(1, 3, 64, 64)
        trace = trace_attention(loaded, img)
        assert trace.indices.shape == (1, 8)


# ===================================================================
# Override tests
# ===================================================================

class TestOverride:
    def test_override_changes_policy(self, tmp_path):
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="sample")
        cfg = ExperimentConfig(
            perception=PerceptionConfig(tokenizer_eval_mode="sample"),
        )
        path = save_checkpoint(tmp_path / "ckpt", model_state=model.state_dict(), config=cfg)
        ckpt = load_checkpoint(path)
        # Override to mean
        loaded = model_from_checkpoint(ckpt, action_dim=3, tokenizer_eval_mode_override="mean")
        assert loaded._tokenizer_eval_mode == "mean"
        assert loaded.tokenizer.eval_mode == "mean"

    def test_no_override_preserves_saved(self, tmp_path):
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        cfg = ExperimentConfig(
            perception=PerceptionConfig(tokenizer_eval_mode="mean"),
        )
        path = save_checkpoint(tmp_path / "ckpt", model_state=model.state_dict(), config=cfg)
        ckpt = load_checkpoint(path)
        loaded = model_from_checkpoint(ckpt, action_dim=3)
        assert loaded._tokenizer_eval_mode == "mean"

    def test_legacy_checkpoint_override_to_mean(self, tmp_path):
        """Legacy bare state_dict defaults to sample, but override can change it."""
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="sample")
        torch.save(model.state_dict(), tmp_path / "legacy.pt")
        ckpt = load_checkpoint(tmp_path / "legacy.pt")
        loaded = model_from_checkpoint(ckpt, action_dim=3, tokenizer_eval_mode_override="mean")
        assert loaded._tokenizer_eval_mode == "mean"

    def test_override_validation_rejects_invalid(self):
        ckpt = {"model_state": ReducedWorldModel(action_dim=3).state_dict()}
        with pytest.raises(AssertionError):
            model_from_checkpoint(ckpt, action_dim=3, tokenizer_eval_mode_override="invalid")

    def test_mean_eval_identical_across_seeds(self, tmp_path):
        """Same checkpoint + mean mode gives identical predictions with different RNG
        seeds (because mean mode uses no noise)."""
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="mean")
        cfg = ExperimentConfig(
            perception=PerceptionConfig(tokenizer_eval_mode="mean"),
        )
        path = save_checkpoint(tmp_path / "ckpt", model_state=model.state_dict(), config=cfg)
        ckpt = load_checkpoint(path)

        img = torch.randn(1, 3, 64, 64)
        act = torch.zeros(1, 3)

        predictions = []
        for seed in (0, 1):
            torch.manual_seed(seed)
            loaded = model_from_checkpoint(ckpt, action_dim=3,
                                           tokenizer_eval_mode_override="mean")
            loaded.eval()
            with torch.no_grad():
                out = loaded(img=img, prev_action=act, current_action=act,
                             force_keep_input=True)
            predictions.append(out.reward_pred)

        torch.testing.assert_close(predictions[0], predictions[1])

    def test_sample_eval_can_vary_across_seeds(self, tmp_path):
        """Same checkpoint + sample mode can produce different results with different seeds."""
        model = ReducedWorldModel(action_dim=3, tokenizer_eval_mode="sample")
        cfg = ExperimentConfig(
            perception=PerceptionConfig(tokenizer_eval_mode="sample"),
        )
        path = save_checkpoint(tmp_path / "ckpt", model_state=model.state_dict(), config=cfg)
        ckpt = load_checkpoint(path)

        predictions = []
        for seed in (0, 1):
            torch.manual_seed(seed)
            loaded = model_from_checkpoint(ckpt, action_dim=3,
                                           tokenizer_eval_mode_override="sample")
            loaded.eval()
            img = torch.randn(1, 3, 64, 64)
            act = torch.zeros(1, 3)
            with torch.no_grad():
                out = loaded(img=img, prev_action=act, current_action=act,
                             force_keep_input=True)
            predictions.append(out.reward_pred)

        # Extremely likely to differ with different RNG seeds
        assert not torch.allclose(predictions[0], predictions[1]), (
            "Sample eval should vary across RNG seeds"
        )
