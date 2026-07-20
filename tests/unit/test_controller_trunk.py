"""Tests for configurable reward head in ControllerTrunk (Stage 2.5C.0)."""

import importlib.util
from pathlib import Path

import pytest
import torch

from rwm.models.controller_trunk import ControllerTrunk
from rwm.models.rwm.model import ReducedWorldModel
from rwm.config.experiment_config import ControllerConfig, ExperimentConfig, DataConfig, TemporalConfig, TrainingConfig
from rwm.trainers.deterministic.world_model_trainer import WorldModelTrainer
from rwm.utils.checkpointing import save_checkpoint, load_checkpoint, model_from_checkpoint


# ===================================================================
# Forward shape and behavior
# ===================================================================

class TestForwardShapes:
    def test_linear_head_shape(self):
        trunk = ControllerTrunk(action_dim=3)
        B, D = 4, 80
        belief = torch.randn(B, D)
        action = torch.randn(B, 3)
        h, r = trunk(belief, action)
        assert h.shape == (B, 80)
        assert r.shape == (B, 1)

    def test_nonlinear_head_shape(self):
        trunk = ControllerTrunk(action_dim=3, reward_head_kind="nonlinear",
                                reward_head_hidden_dim=64)
        B, D = 4, 80
        belief = torch.randn(B, D)
        action = torch.randn(B, 3)
        h, r = trunk(belief, action)
        assert h.shape == (B, 80)
        assert r.shape == (B, 1)

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="reward_head_kind"):
            ControllerTrunk(action_dim=3, reward_head_kind="invalid")

    def test_invalid_hidden_dim_raises_at_construction(self):
        with pytest.raises(ValueError, match="reward_head_hidden_dim"):
            ControllerTrunk(action_dim=3, reward_head_kind="nonlinear",
                            reward_head_hidden_dim=0)

    def test_gradients_flow_to_belief_and_action(self):
        trunk = ControllerTrunk(action_dim=3, reward_head_kind="nonlinear")
        belief = torch.randn(2, 80, requires_grad=True)
        action = torch.randn(2, 3, requires_grad=True)
        _, r = trunk(belief, action)
        loss = r.sum()
        loss.backward()
        assert belief.grad is not None and belief.grad.abs().sum().item() > 0
        assert action.grad is not None and action.grad.abs().sum().item() > 0


# ===================================================================
# Full model integration
# ===================================================================

class TestModelIntegration:
    def test_linear_model_forward(self):
        model = ReducedWorldModel(action_dim=3, reward_head_kind="linear")
        model.eval()
        img = torch.randn(1, 3, 64, 64)
        act = torch.zeros(1, 3)
        out = model(img=img, prev_action=act, current_action=act, force_keep_input=True)
        assert out.reward_pred.shape == (1, 1)

    def test_nonlinear_model_forward(self):
        model = ReducedWorldModel(action_dim=3, reward_head_kind="nonlinear")
        model.eval()
        img = torch.randn(1, 3, 64, 64)
        act = torch.zeros(1, 3)
        out = model(img=img, prev_action=act, current_action=act, force_keep_input=True)
        assert out.reward_pred.shape == (1, 1)

    def test_nonlinear_model_gradients(self):
        model = ReducedWorldModel(action_dim=3, reward_head_kind="nonlinear")
        model.train()
        img = torch.randn(2, 3, 64, 64)
        act = torch.randn(2, 3)
        out = model(img=img, prev_action=act, current_action=act, force_keep_input=True)
        loss = out.reward_pred.sum()
        loss.backward()
        # Verify gradients reach the controller's reward head
        has_grad = False
        for p in model.controller.reward_head.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                has_grad = True
                break
        assert has_grad


# ===================================================================
# Checkpoint compatibility
# ===================================================================

class TestCheckpointCompat:
    def _make_linear_state_dict(self):
        model = ReducedWorldModel(action_dim=3, reward_head_kind="linear")
        return model.state_dict()

    def _make_nonlinear_state_dict(self):
        model = ReducedWorldModel(action_dim=3, reward_head_kind="nonlinear")
        return model.state_dict()

    def test_legacy_checkpoint_loads_as_linear(self, tmp_path):
        """A legacy bare state_dict (no config) must load into a linear model."""
        state = self._make_linear_state_dict()
        path = tmp_path / "legacy.pt"
        torch.save(state, path)

        loaded = load_checkpoint(path)
        assert loaded["legacy"]
        assert loaded["config"] is None

        model = model_from_checkpoint(loaded, action_dim=3)
        # Verify it's linear (single Linear layer, not Sequential)
        assert isinstance(model.controller.reward_head, torch.nn.Linear)

    def test_linear_and_nonlinear_state_dicts_differ(self):
        """The state dict keys must differ between linear and nonlinear heads."""
        lin = self._make_linear_state_dict()
        nonlin = self._make_nonlinear_state_dict()
        # Both should have controller.reward_head.weight, but the nonlinear
        # version should also have controller.reward_head.0.weight etc.
        lin_keys = set(lin.keys())
        nonlin_keys = set(nonlin.keys())
        # Nonlinear has deeper structure
        assert lin_keys != nonlin_keys
        assert "controller.reward_head.weight" in lin_keys
        assert "controller.reward_head.0.weight" in nonlin_keys

    def test_structured_nonlinear_checkpoint_roundtrip(self, tmp_path):
        """Save a nonlinear model, reload, and verify identical eval output."""
        import copy
        model1 = ReducedWorldModel(action_dim=3, reward_head_kind="nonlinear",
                                   reward_head_hidden_dim=64,
                                   tokenizer_eval_mode="mean")
        model1.eval()

        from rwm.config.experiment_config import PerceptionConfig
        cfg = ControllerConfig(reward_head_kind="nonlinear", reward_head_hidden_dim=64)
        ckpt_path = save_checkpoint(
            tmp_path / "nonlinear",
            model_state=model1.state_dict(),
            config=__import__("rwm.config.experiment_config",
                              fromlist=["ExperimentConfig"]).ExperimentConfig(
                controller=cfg,
                perception=PerceptionConfig(tokenizer_eval_mode="mean"),
            ),
        )
        loaded_ckpt = load_checkpoint(ckpt_path)
        model2 = model_from_checkpoint(loaded_ckpt, action_dim=3)
        model2.eval()

        # Same input → same output
        img = torch.randn(1, 3, 64, 64)
        act = torch.zeros(1, 3)
        with torch.no_grad():
            o1 = model1(img=img, prev_action=act, current_action=act, force_keep_input=True)
            o2 = model2(img=img, prev_action=act, current_action=act, force_keep_input=True)
        torch.testing.assert_close(o1.reward_pred, o2.reward_pred)


# ===================================================================
# Config serialization
# ===================================================================

class TestConfig:
    def test_controller_config_defaults(self):
        cfg = ControllerConfig()
        assert cfg.reward_head_kind == "linear"
        assert cfg.reward_head_hidden_dim == 32

    def test_controller_config_round_trip(self):
        cfg = ControllerConfig(reward_head_kind="nonlinear", reward_head_hidden_dim=128)
        d = cfg.to_dict()
        loaded = ControllerConfig.from_dict(d)
        assert loaded.reward_head_kind == "nonlinear"
        assert loaded.reward_head_hidden_dim == 128


# ===================================================================
# Trainer integration tests
# ===================================================================

class TestTrainerIntegration:
    def test_trainer_uses_config_nonlinear_head(self, tmp_path):
        """Trainer created with nonlinear ExperimentConfig must have a nonlinear head."""
        cfg = ExperimentConfig(
            experiment_name="test",
            controller=ControllerConfig(reward_head_kind="nonlinear", reward_head_hidden_dim=64),
            training=TrainingConfig(batch_size=2),
        )
        from torch.utils.data import DataLoader
        from rwm.data.rollout_dataset import RolloutDataset
        import numpy as np
        p = tmp_path / "d"
        p.mkdir()
        np.savez_compressed(p / "ep.npz", obs=np.zeros((20, 8, 8, 3), dtype=np.uint8),
                            action=np.zeros((20, 3)), reward=np.zeros(20), done=np.zeros(20, dtype=bool))
        ds = RolloutDataset(root_dir=p, sequence_len=8, image_size=8)
        loader = DataLoader(ds, batch_size=2, shuffle=False, drop_last=True)
        trainer = WorldModelTrainer(train_loader=loader, out_dir=tmp_path / "out",
                                    sequence_len=8, epochs=1, batch_size=2, config=cfg)
        assert trainer.model._reward_head_kind == "nonlinear"
        assert isinstance(trainer.model.controller.reward_head, torch.nn.Sequential)

    def test_trainer_defaults_linear(self, tmp_path):
        """Trainer without config defaults to linear head."""
        from torch.utils.data import DataLoader
        from rwm.data.rollout_dataset import RolloutDataset
        import numpy as np
        p = tmp_path / "d2"
        p.mkdir()
        np.savez_compressed(p / "ep.npz", obs=np.zeros((20, 8, 8, 3), dtype=np.uint8),
                            action=np.zeros((20, 3)), reward=np.zeros(20), done=np.zeros(20, dtype=bool))
        ds = RolloutDataset(root_dir=p, sequence_len=8, image_size=8)
        loader = DataLoader(ds, batch_size=2, shuffle=False, drop_last=True)
        trainer = WorldModelTrainer(train_loader=loader, out_dir=tmp_path / "out2",
                                    sequence_len=8, epochs=1, batch_size=2)
        assert trainer.model._reward_head_kind == "linear"

    def test_model_from_checkpoint_bad_kind_raises(self):
        """Malformed metadata must raise ValueError, not fall back silently."""
        cfg_dict = {"controller": {"reward_head_kind": "invalid_kind", "reward_head_hidden_dim": 64}}
        ckpt = {"config": cfg_dict, "model_state": {}}
        with pytest.raises(ValueError, match="Invalid reward_head_kind"):
            model_from_checkpoint(ckpt)

    def test_model_from_checkpoint_bad_hidden_raises(self):
        """Non-positive hidden dim must raise ValueError."""
        cfg_dict = {"controller": {"reward_head_kind": "nonlinear", "reward_head_hidden_dim": 0}}
        ckpt = {"config": cfg_dict, "model_state": {}}
        with pytest.raises(ValueError, match="Invalid reward_head_hidden_dim"):
            model_from_checkpoint(ckpt)

    def test_action_probe_nonlinear_checkpoint(self, tmp_path):
        """Action probe must load a nonlinear checkpoint without crashing."""
        script_path = (
            Path(__file__).parents[2] / "scripts" / "evaluation"
            / "evaluate_reward_prediction.py"
        )
        spec = importlib.util.spec_from_file_location("evaluate_reward_prediction", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model = ReducedWorldModel(action_dim=3, reward_head_kind="nonlinear")
        model.eval()
        cfg = ExperimentConfig(
            experiment_name="probe_test",
            controller=ControllerConfig(reward_head_kind="nonlinear"),
        )
        ckpt_path = save_checkpoint(tmp_path / "nonlinear_ckpt", model_state=model.state_dict(), config=cfg)
        result = module.action_probe(ckpt_path, seed=7)
        assert result["unique"] >= 1

    def test_linear_checkpoint_still_loads(self, tmp_path):
        """Existing legacy linear checkpoint must still load correctly."""
        model = ReducedWorldModel(action_dim=3, reward_head_kind="linear")
        model.eval()
        torch.save(model.state_dict(), tmp_path / "legacy.pt")
        ckpt = load_checkpoint(tmp_path / "legacy.pt")
        loaded = model_from_checkpoint(ckpt, action_dim=3)
        loaded.eval()
        assert isinstance(loaded.controller.reward_head, torch.nn.Linear)
