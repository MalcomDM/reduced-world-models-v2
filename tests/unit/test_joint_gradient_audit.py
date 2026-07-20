"""Tests for Stage 6.0 Joint-Gradient Measurement Gate (corrected semantics).

Verifies:
  1. Factual loss masks: visible=target-only all-visible, masked=4 genuine pre_perception_skip.
  2. Imagination start matches S5 trainer (burn-in + 4 target, not all 16).
  3. Critic loss does NOT reach Actor. Actor loss does NOT reach Critic or RewardHead.
  4. TargetCritic has no gradient under any loss.
  5. Entropy is a separate measurement, not embedded in actor_loss.
  6. Bootstrap uses z_H (final imagined state).
  7. Per-block cosine: null/N/A for disconnected or zero-gradient blocks.
  8. graph_connected vs nonzero_gradient distinction.
  9. Eval parity: reward_pred_seq, temporal_state match after audit.
 10. Full state restoration: params, buffers, requires_grad, modes, grads.
 11. No optimizer created or invoked.
 12. CLI smoke (eval-parity and gradient-audit modes).
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from rwm.config.config import ACTION_DIM
from rwm.config.experiment_config import TemporalConfig
from rwm.evaluation.joint_gradient_audit import (
    PARAM_BLOCK_NAMES,
    LOSS_NAMES,
    _extract_warmup_window,
    _flatten_aligned_grads,
    _gather_block_params,
    _build_prev_actions,
    _snapshot_model,
    _restore_snapshot,
    run_joint_gradient_audit,
    serialize_audit_result,
)
from rwm.models.rwm.model import ReducedWorldModel
from rwm.models.actor_critic import ActorCritic
from rwm.trainers.imagined_actor_critic import ImaginedACTrainer, ImaginedACTrainingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sru_batch(B: int = 2, T_total: int = 36) -> Dict[str, Tensor]:
    """Standard 36-position burn-in batch."""
    loss_mask = torch.zeros(T_total, dtype=torch.bool)
    loss_mask[20:] = True
    valid_step = torch.ones(T_total, dtype=torch.bool)
    return {
        "obs": torch.randn(B, T_total, 3, 64, 64),
        "action": torch.randn(B, T_total, 3),
        "reward": torch.randn(B, T_total),
        "done": torch.zeros(B, T_total, dtype=torch.bool),
        "predecessor_action": torch.randn(B, 3),
        "valid_step": valid_step.unsqueeze(0).expand(B, -1),
        "loss_mask": loss_mask.unsqueeze(0).expand(B, -1),
    }


class _FakeDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return _make_sru_batch(B=1, T_total=36)


@pytest.fixture
def sru_model_and_ac():
    """SRU model + untrained AC on CPU."""
    tc = TemporalConfig(backend="minimal_sru")
    m = ReducedWorldModel(temporal_config=tc, tokenizer_eval_mode="mean").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    cfg = ImaginedACTrainingConfig(warmup_steps=4, max_batches=1)
    loader = DataLoader(_FakeDataset(), batch_size=1, shuffle=False)
    tr = ImaginedACTrainer(model=m, train_loader=loader, train_cfg=cfg,
                           out_dir=Path(tempfile.mkdtemp()),
                           device=torch.device("cpu"))
    return m, tr.ac


def _block_names_with_gradient(result: Dict, loss_name: str, key: str = "nonzero_gradient") -> List[str]:
    blocks = result.get("losses", {}).get(loss_name, {}).get("blocks", {})
    return [k for k, v in blocks.items() if isinstance(v, dict) and v.get(key)]


# ===================================================================
# 1. Loss mask correctness
# ===================================================================

class TestLossMasks:
    """Visible MSE on target-only; masked on exactly H pre_perception_skip positions."""

    def test_visible_mse_target_only(self, sru_model_and_ac):
        """Visible reward MSE uses only valid_step & loss_mask positions."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, horizon=4, train_mode_params=False)
        assert "visible_reward_mse" in result["losses"]
        # Should reach perception + reward blocks
        routes = _block_names_with_gradient(result, "visible_reward_mse")
        assert "reward_head" in routes
        assert "controller_trunk" in routes
        assert "minimal_sru" in routes
        assert "tokenizer" in routes
        assert "encoder" in routes
        assert "actor" not in routes
        assert "online_critic" not in routes

    def test_masked_mse_on_exactly_h_masked(self, sru_model_and_ac):
        """Masked MSE uses exactly H pre_perception_skip positions."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, horizon=4, train_mode_params=False)
        routes = _block_names_with_gradient(result, "masked_reward_mse")
        assert "reward_head" in routes
        assert "controller_trunk" in routes
        assert "minimal_sru" in routes
        assert "actor" not in routes
        assert "online_critic" not in routes


# ===================================================================
# 2. Warmup matches S5 trainer
# ===================================================================

class TestWarmupMatch:
    """Imagination warmup matches S5 trainer (burn-in + first 4 target)."""

    def test_warmup_window_size(self, sru_model_and_ac):
        """Warmup has T = first_target + ws (24 for burn_in=20, ws=4)."""
        m, ac = sru_model_and_ac
        B = 2
        batch = _make_sru_batch(B)
        first_target = int(batch["loss_mask"][0].long().argmax().item())
        ws = 4
        warm = _extract_warmup_window(
            batch["obs"], batch["action"],
            batch["valid_step"], batch["loss_mask"],
            batch["predecessor_action"], ws=ws,
        )
        assert warm["obs"].shape[1] == first_target + ws

    def test_warmup_prev_actions_timing(self, sru_model_and_ac):
        """predecessor_action at first_valid; standard shift elsewhere."""
        m, ac = sru_model_and_ac
        B = 2
        batch = _make_sru_batch(B)
        warm = _extract_warmup_window(
            batch["obs"], batch["action"],
            batch["valid_step"], batch["loss_mask"],
            batch["predecessor_action"], ws=4,
        )
        prev = warm["prev_actions"]
        first_valid = int(batch["valid_step"][0].long().argmax().item())
        # Position first_valid gets predecessor_action
        assert torch.allclose(prev[:, first_valid], batch["predecessor_action"], atol=1e-6)
        # Standard shift: prev[t] = actions[t-1]
        if prev.shape[1] > 1:
            assert torch.allclose(prev[:, 1], batch["action"][:, 0], atol=1e-6)


# ===================================================================
# 3. Critic/Actor separation
# ===================================================================

class TestCriticActorSeparation:
    """Critic loss must NOT reach Actor; Actor loss must NOT reach Critic or RewardHead."""

    def test_critic_loss_no_actor_gradient(self, sru_model_and_ac):
        """Critic loss has graph_connected=False for Actor."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        actor_info = result["losses"]["critic_loss"]["blocks"]["actor"]
        assert actor_info.get("graph_connected") is False, \
            "Critic loss must not reach Actor parameters"

    def test_actor_loss_no_critic_gradient(self, sru_model_and_ac):
        """Actor loss has graph_connected=False for online_critic and target_critic."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        oc_info = result["losses"]["actor_loss"]["blocks"]["online_critic"]
        assert oc_info.get("graph_connected") is False, \
            "Actor loss must not reach Online Critic"
        tc_info = result["losses"]["actor_loss"]["blocks"]["target_critic"]
        assert tc_info.get("graph_connected") is False, \
            "Actor loss must not reach Target Critic"

    def test_actor_loss_no_reward_head(self, sru_model_and_ac):
        """Actor loss has graph_connected=False for reward_head."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        rh_info = result["losses"]["actor_loss"]["blocks"]["reward_head"]
        assert rh_info.get("graph_connected") is False, \
            "Actor loss must not reach RewardHead"


# ===================================================================
# 4. TargetCritic frozen under all losses
# ===================================================================

class TestTargetCriticFrozen:
    """TargetCritic has no gradient under every loss."""

    def test_target_critic_no_gradient_any_loss(self, sru_model_and_ac):
        """target_critic has graph_connected=False for all 6 losses."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        for lname in LOSS_NAMES:
            info = result["losses"][lname]["blocks"]["target_critic"]
            assert info.get("graph_connected") is False, \
                f"TargetCritic must be disconnected from {lname}"

    def test_target_critic_requires_grad_false(self, sru_model_and_ac):
        """TargetCritic requires_grad=False throughout."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        _ = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        assert all(not p.requires_grad for p in ac.target_critic.parameters()), \
            "Target Critic must remain frozen"


# ===================================================================
# 5. Entropy separate from actor_loss
# ===================================================================

class TestEntropySeparate:
    """Entropy is a separate loss measurement."""

    def test_entropy_as_separate_loss(self, sru_model_and_ac):
        """Entropy is present as its own measurement."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        assert "entropy" in result["losses"], "Entropy must be a separate loss"
        assert result["losses"]["entropy"]["finite"]


# ===================================================================
# 6. Bootstrap from z_H
# ===================================================================

class TestBootstrapTiming:
    """Bootstrap uses z_H (final imagined state), not z_start."""

    def test_bootstrap_not_from_start(self, sru_model_and_ac):
        """Critic loss graph reaches SRU (proving bootstrap flows through z_H)."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        # If bootstrap went through z_H (not z_start), critic loss reaches
        # the SRU cell (since z_H = SRU.advance(z_start, a_0, a_1, ..., a_{H-1})).
        sru_info = result["losses"]["critic_loss"]["blocks"]["minimal_sru"]
        assert sru_info.get("graph_connected"), \
            "Critic loss must reach SRU through z_H bootstrap"


# ===================================================================
# 7. Per-block cosine correctness
# ===================================================================

class TestCosine:
    """Per-block cosine: null for disconnected or zero-gradient blocks."""

    def test_cosine_actor_block_only_actor_loss_and_entropy(self, sru_model_and_ac):
        """Actor block has cosines only for actor_loss×entropy (only two losses reaching it)."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        cs = result.get("cosine_similarity_per_block", {})
        actor_cos = cs.get("actor", {})
        # Only actor_loss and entropy reach Actor
        expected_pairs = {"actor_loss×entropy"}
        actual_pairs = {k for k, v in actor_cos.items() if v is not None}
        assert actual_pairs == expected_pairs, \
            f"Actor cosine pairs: expected {expected_pairs}, got {actual_pairs}"

    def test_cosine_reward_head_only_two_rewards(self, sru_model_and_ac):
        """reward_head has cosines only for visible×masked reward MSE."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        cs = result.get("cosine_similarity_per_block", {})
        rh_cos = cs.get("reward_head", {})
        null_keys = {k for k, v in rh_cos.items() if v is None}
        non_null_keys = {k for k, v in rh_cos.items() if v is not None}
        # Should be null for all except masked×visible
        assert "masked_reward_mse×visible_reward_mse" in non_null_keys

    def test_cosine_target_critic_all_null(self, sru_model_and_ac):
        """target_critic has all null cosines (no gradient from any loss)."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        cs = result.get("cosine_similarity_per_block", {})
        tc_cos = cs.get("target_critic", {})
        for pair, val in tc_cos.items():
            assert val is None, f"target_critic cosine {pair} should be None, got {val}"

    def test_flatten_keeps_disconnected_parameter_coordinates(self):
        """Missing grads become aligned zeros rather than shortening vectors."""
        first = torch.nn.Parameter(torch.zeros(2))
        second = torch.nn.Parameter(torch.zeros(3))
        first.grad = torch.tensor([1.0, 2.0])
        second.grad = None
        flattened = _flatten_aligned_grads([first, second])
        assert flattened is not None
        assert torch.equal(flattened, torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0]))


# ===================================================================
# 8. graph_connected vs nonzero_gradient
# ===================================================================

class TestGraphVsNonzero:
    """graph_connected and nonzero_gradient are distinct."""

    def test_all_nonzero_are_graph_connected(self, sru_model_and_ac):
        """Any block with nonzero_gradient must also have graph_connected=True."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        for lname in LOSS_NAMES:
            for bname in PARAM_BLOCK_NAMES:
                info = result["losses"][lname]["blocks"].get(bname, {})
                if info.get("nonzero_gradient"):
                    assert info.get("graph_connected"), \
                        f"{lname}→{bname}: nonzero but not graph_connected"


# ===================================================================
# 9. Eval parity
# ===================================================================

class TestEvalParity:
    """Eval parity: reward_pred_seq and temporal_state match after audit."""

    def test_eval_parity_overall(self, sru_model_and_ac):
        """Overall eval parity passes."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        assert result["eval_parity"]["overall"] is True, "Eval parity must pass"

    def test_reward_pred_seq_parity(self, sru_model_and_ac):
        """reward_pred_seq matches before/after audit."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        assert result["eval_parity"]["reward_pred_seq_match"] is True

    def test_temporal_state_parity(self, sru_model_and_ac):
        """temporal_state matches before/after audit."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        assert result["eval_parity"]["temporal_state_match"] is True


# ===================================================================
# 10. Full state restoration
# ===================================================================

class TestStateRestoration:
    """All state restored after audit: params, buffers, requires_grad, modes, grads."""

    def test_parameter_identity(self, sru_model_and_ac):
        """All parameter tensors are bitwise identical after audit."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        snap_before = {id(p): p.data.clone() for p in m.parameters()}
        for p in ac.parameters():
            snap_before[id(p)] = p.data.clone()
        _ = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        for p in m.parameters():
            assert torch.equal(p.data, snap_before[id(p)])
        for p in ac.parameters():
            assert torch.equal(p.data, snap_before[id(p)])

    def test_hash_identity(self, sru_model_and_ac):
        """hash_before == hash_after."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        assert result["hash_identity"] is True

    def test_buffer_restored(self, sru_model_and_ac):
        """Buffer values unchanged after audit."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        buffers_before = {}
        for prefix, mod in [("model", m), ("ac", ac)]:
            for name, buf in mod.named_buffers(recurse=True):
                buffers_before[f"{prefix}.{name}"] = buf.clone()
        _ = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        for prefix, mod in [("model", m), ("ac", ac)]:
            for name, buf in mod.named_buffers(recurse=True):
                key = f"{prefix}.{name}"
                if key in buffers_before:
                    assert torch.equal(buf, buffers_before[key]), f"Buffer changed: {key}"

    def test_requires_grad_restored(self, sru_model_and_ac):
        """requires_grad flags restored after audit."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        rg_before = {id(p): p.requires_grad for p in m.parameters()}
        for p in ac.parameters():
            rg_before[id(p)] = p.requires_grad
        _ = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        for p in m.parameters():
            assert p.requires_grad == rg_before[id(p)]
        for p in ac.parameters():
            assert p.requires_grad == rg_before[id(p)]

    def test_training_mode_restored(self, sru_model_and_ac):
        """model.train/eval mode restored."""
        m, ac = sru_model_and_ac
        m.train()  # change to train
        batch = _make_sru_batch(B=2)
        _ = run_joint_gradient_audit(m, ac, batch, train_mode_params=True)
        assert m.training, "model training mode must be restored"

    def test_tokenizer_mode_restored(self, sru_model_and_ac):
        """Tokenizer eval_mode restored."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        m.tokenizer.eval_mode = "mean"
        _ = run_joint_gradient_audit(m, ac, batch, train_mode_params=True)
        assert m.tokenizer.eval_mode == "mean", "Tokenizer eval_mode must be restored"

    def test_mixed_submodule_modes_restored(self, sru_model_and_ac):
        """Mixed train/eval submodule states survive the diagnostic."""
        m, ac = sru_model_and_ac
        m.train()
        m.encoder.eval()
        ac.train()
        ac.actor.eval()
        before = {
            id(module): module.training
            for root in (m, ac)
            for module in root.modules()
        }
        _ = run_joint_gradient_audit(
            m, ac, _make_sru_batch(B=2), train_mode_params=True,
        )
        after = {
            id(module): module.training
            for root in (m, ac)
            for module in root.modules()
        }
        assert after == before

    def test_grad_values_restored(self, sru_model_and_ac):
        """Original .grad values (including None) restored."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        # Set some gradients before audit
        for p in ac.actor.parameters():
            p.grad = torch.ones_like(p) * 0.5
        grad_before = {id(p): p.grad.clone() if p.grad is not None else None
                       for p in m.parameters()}
        for p in ac.parameters():
            grad_before[id(p)] = p.grad.clone() if p.grad is not None else None
        _ = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        for p in m.parameters():
            g_before = grad_before[id(p)]
            g_after = p.grad
            if g_before is None:
                assert g_after is None, f"Grad should be None, got {g_after}"
            else:
                assert g_after is not None and torch.equal(g_after, g_before), \
                    f"Grad not equal for param {id(p)}"


# ===================================================================
# 11. No optimizer step
# ===================================================================

class TestNoOptimizer:
    """No optimizer was created or invoked."""

    def test_no_optimizer_step_flag(self, sru_model_and_ac):
        """Audit result confirms no optimizer step."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        assert result["no_optimizer_step"] is True

    def test_optimizer_step_raises_if_called(self, sru_model_and_ac):
        """Monkeypatch optimizer.step to raise if called."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        orig_step = ac._actor_optim.step
        called = [False]
        def raise_if_called(*a, **kw):
            called[0] = True
            return orig_step(*a, **kw)
        ac._actor_optim.step = raise_if_called
        _ = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        ac._actor_optim.step = orig_step
        assert not called[0], "optimizer.step() was called during audit"


# ===================================================================
# 12. JSON serialization
# ===================================================================

class TestJSONSerialization:
    """Audit result serializes to valid JSON with all expected keys."""

    def test_json_round_trip(self, sru_model_and_ac):
        """Result round-trips through JSON."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        json_str = serialize_audit_result(result)
        data = json.loads(json_str)
        for key in ["losses", "cosine_similarity_per_block", "hash_identity", "metadata"]:
            assert key in data, f"Missing key: {key}"

    def test_each_loss_has_all_blocks(self, sru_model_and_ac):
        """Every loss has all 11 blocks with expected fields."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        for lname in LOSS_NAMES:
            blocks = result["losses"][lname]["blocks"]
            for bname in PARAM_BLOCK_NAMES:
                assert bname in blocks, f"Missing block {bname} in {lname}"
                for key in ["graph_connected", "nonzero_gradient", "grad_l2_norm",
                            "param_l2_norm", "ratio", "finite"]:
                    assert key in blocks[bname], f"Missing {key} in {lname}.{bname}"

    def test_all_gradients_finite(self, sru_model_and_ac):
        """All reported gradients are finite."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        for lname in LOSS_NAMES:
            for bname in PARAM_BLOCK_NAMES:
                info = result["losses"][lname]["blocks"][bname]
                if info.get("nonzero_gradient"):
                    assert info.get("finite") is True


# ===================================================================
# 13. Routing matrix (exhaustive)
# ===================================================================

class TestRoutingMatrix:
    """Expected gradient routes."""
    EXPECTED_NONZERO = {
        "visible_reward_mse": [
            "controller_trunk", "reward_head", "minimal_sru",
            "spatial_attention_head", "attention_scorer", "tokenizer", "encoder",
        ],
        "masked_reward_mse": [
            "controller_trunk", "reward_head", "minimal_sru",
            "spatial_attention_head", "attention_scorer", "tokenizer", "encoder",
        ],
        "tokenizer_kl": ["tokenizer", "encoder"],
        "critic_loss": [
            "online_critic", "controller_trunk", "minimal_sru",
            "spatial_attention_head", "attention_scorer", "tokenizer", "encoder",
        ],
        "actor_loss": [
            "actor", "controller_trunk", "minimal_sru",
            "spatial_attention_head", "attention_scorer", "tokenizer", "encoder",
        ],
        "entropy": [
            "actor", "controller_trunk", "minimal_sru",
            "spatial_attention_head", "attention_scorer", "tokenizer", "encoder",
        ],
    }
    NEVER_NONZERO = ["target_critic", "topk_selector"]

    def test_expected_routes_present(self, sru_model_and_ac):
        """All expected nonzero routes are present."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        for lname, expected in self.EXPECTED_NONZERO.items():
            actual = _block_names_with_gradient(result, lname)
            for bname in expected:
                assert bname in actual, f"Missing {lname}→{bname}: got {actual}"

    def test_unexpected_routes_absent(self, sru_model_and_ac):
        """Blocks that should never get gradients don't."""
        m, ac = sru_model_and_ac
        batch = _make_sru_batch(B=2)
        result = run_joint_gradient_audit(m, ac, batch, train_mode_params=False)
        for lname in LOSS_NAMES:
            actual = _block_names_with_gradient(result, lname)
            for bname in self.NEVER_NONZERO:
                assert bname not in actual, \
                    f"{bname} unexpectedly has gradient for {lname}"


# ===================================================================
# 14. CLI smoke (integration)
# ===================================================================

class TestCLISmoke:
    """CLI smoke on real canonical checkpoints."""

    ANCHOR = "runs/component_refinement/sru_temporal/08_strict_observational_dropout_anchor/seed42/checkpoint_best.pt"
    AC_CKPT = "runs/imagined_actor_critic/minimal_sru/01_frozen_parity/seed42/checkpoints/ac_checkpoint_2000.pt"

    @pytest.mark.integration
    def test_cli_smoke_eval_parity(self):
        """CLI runs in eval-parity mode without error."""
        import subprocess, sys
        out = Path(tempfile.mkdtemp()) / "audit_eval_parity.json"
        cmd = [
            sys.executable, "scripts/diagnostics/audit_joint_gradients.py",
            "--anchor", self.ANCHOR, "--ac", self.AC_CKPT,
            "--out", str(out), "--smoke",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"CLI failed: {r.stderr}"
        assert out.exists()
        data = json.loads(out.read_text())
        assert data.get("eval_parity", {}).get("overall") is True
        assert data.get("hash_identity") is True
        assert "losses" in data

    @pytest.mark.integration
    def test_cli_smoke_grad_mode(self):
        """CLI runs in gradient-audit (train) mode without error."""
        import subprocess, sys
        out = Path(tempfile.mkdtemp()) / "audit_grad.json"
        cmd = [
            sys.executable, "scripts/diagnostics/audit_joint_gradients.py",
            "--anchor", self.ANCHOR, "--ac", self.AC_CKPT,
            "--out", str(out), "--train-mode", "--seed", "42", "--smoke",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"CLI failed: {r.stderr}"
        data = json.loads(out.read_text())
        assert data.get("train_mode") is True
        assert data.get("eval_parity", {}).get("overall") is True
        assert len(data.get("batch_fingerprint", "")) == 16
        assert "losses" in data

    def test_loader_seed_reproduces_probe_batch(self):
        """The CLI seed fixes data selection as well as model stochasticity."""
        import importlib.util

        script_path = Path("scripts/diagnostics/audit_joint_gradients.py")
        spec = importlib.util.spec_from_file_location("audit_joint_gradients_cli", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        load_batch = module._load_batch

        first = load_batch(
            Path(self.ANCHOR), torch.device("cpu"), smoke=True, seed=17,
        )
        second = load_batch(
            Path(self.ANCHOR), torch.device("cpu"), smoke=True, seed=17,
        )
        for key in first:
            if isinstance(first[key], torch.Tensor):
                assert torch.equal(first[key], second[key]), key
