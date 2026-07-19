"""Tests for Top-K selection modes (Stage 2.5C.2A).

Modes: learned, fixed_uniform, fixed_random.
"""

import pytest
import torch

from rwm.models.rwm.topk_gumbel_selector import (
    TopKGumbelSelector,
    _farthest_point_sample,
    _fixed_random_indices,
    _N_PATCHES,
)
from rwm.models.rwm.spatial_attention_head import SpatialAttentionHead
from rwm.models.rwm.model import ReducedWorldModel
from rwm.evaluation.attention_trace import trace_attention
from rwm.utils.checkpointing import save_checkpoint, load_checkpoint, model_from_checkpoint
from rwm.config.experiment_config import ExperimentConfig, PerceptionConfig

_KS = (4, 8, 16, 32)
from rwm.models.rwm.spatial_attention_head import SpatialAttentionHead
from rwm.config.config import K


# ===================================================================
# Farthest-point sampling (2-D uniform)
# ===================================================================

class TestFarthestPoint:
    K = 8
    SIDE = 15

    def test_returns_correct_count(self):
        idx = _farthest_point_sample(self.K)
        assert idx.shape == (self.K,)
        assert len(idx.unique()) == self.K

    def test_all_within_range(self):
        idx = _farthest_point_sample(self.K)
        assert (idx >= 0).all() and (idx < 225).all()

    def test_deterministic(self):
        idx1 = _farthest_point_sample(self.K)
        idx2 = _farthest_point_sample(self.K)
        torch.testing.assert_close(idx1, idx2)

    def test_K1_returns_centre(self):
        idx = _farthest_point_sample(1)
        centre = 7 * 15 + 7  # (7, 7) on 15×15
        assert idx[0].item() == centre

    def test_K225_returns_all(self):
        idx = _farthest_point_sample(225)
        assert len(idx.unique()) == 225

    def test_spans_multiple_rows_and_columns(self):
        idx = _farthest_point_sample(self.K)
        rows = idx // self.SIDE
        cols = idx % self.SIDE
        assert rows.unique().numel() >= 2, "Must span at least 2 rows"
        assert cols.unique().numel() >= 2, "Must span at least 2 columns"
        row_span = rows.max().item() - rows.min().item()
        col_span = cols.max().item() - cols.min().item()
        assert row_span >= 7, f"Row span should cover ~half the grid, got {row_span}"
        assert col_span >= 7, f"Col span should cover ~half the grid, got {col_span}"


class TestFixedRandomIndices:
    def test_deterministic(self):
        idx1 = _fixed_random_indices(K, seed=42)
        idx2 = _fixed_random_indices(K, seed=42)
        torch.testing.assert_close(idx1, idx2)

    def test_different_seeds_differ(self):
        idx1 = _fixed_random_indices(K, seed=42)
        idx2 = _fixed_random_indices(K, seed=99)
        assert not torch.allclose(idx1, idx2)


# ===================================================================
# K validation
# ===================================================================

class TestKValidation:
    def test_K0_raises(self):
        with pytest.raises(ValueError, match="k must be"):
            TopKGumbelSelector(k=0)

    def test_K_above_max_raises(self):
        with pytest.raises(ValueError, match="k must be"):
            TopKGumbelSelector(k=300)

    def test_K_float_raises(self):
        with pytest.raises(ValueError, match="k must be"):
            TopKGumbelSelector(k=8.0)

    def test_K1_is_valid(self):
        sel = TopKGumbelSelector(k=1)
        sel.eval()
        mask, idx = sel(torch.randn(2, 225))
        assert idx.shape == (2, 1)
        assert mask.sum(dim=1).unique().tolist() == [1]


# ===================================================================
# Selector mode shapes
# ===================================================================

class TestSelectorModes:
    def test_learned_forward_shape(self):
        sel = TopKGumbelSelector(selection_mode="learned")
        sel.eval()
        logits = torch.randn(4, 225)
        mask, indices = sel(logits)
        assert mask.shape == (4, 225)
        assert indices.shape == (4, K)
        assert mask.sum(dim=1).unique().tolist() == [K]

    def test_fixed_uniform_forward_shape(self):
        sel = TopKGumbelSelector(selection_mode="fixed_uniform")
        sel.eval()
        logits = torch.randn(4, 225)
        mask, indices = sel(logits)
        assert mask.shape == (4, 225)
        assert indices.shape == (4, K)
        assert mask.sum(dim=1).unique().tolist() == [K]

    def test_fixed_uniform_deterministic(self):
        sel = TopKGumbelSelector(selection_mode="fixed_uniform")
        sel.eval()
        logits = torch.randn(2, 225)
        _, idx1 = sel(logits)
        _, idx2 = sel(logits)
        torch.testing.assert_close(idx1, idx2)

    def test_fixed_random_deterministic_eval(self):
        sel = TopKGumbelSelector(selection_mode="fixed_random", selection_seed=42)
        sel.eval()
        logits = torch.randn(2, 225)
        _, idx1 = sel(logits)
        _, idx2 = sel(logits)
        torch.testing.assert_close(idx1, idx2)

    def test_fixed_random_changes_with_seed(self):
        sel1 = TopKGumbelSelector(selection_mode="fixed_random", selection_seed=0)
        sel2 = TopKGumbelSelector(selection_mode="fixed_random", selection_seed=99)
        sel1.eval(); sel2.eval()
        logits = torch.randn(1, 225)
        _, idx1 = sel1(logits)
        _, idx2 = sel2(logits)
        assert not torch.allclose(idx1, idx2)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="selection_mode"):
            TopKGumbelSelector(selection_mode="invalid")


# ===================================================================
# Gradient: learned (STE) and fixed modes
# ===================================================================

class TestGradient:
    def test_learned_unselected_get_gradient(self):
        """Learned mode: unselected scorer logits receive STE gradient
        through the spatial-head path."""
        sel = TopKGumbelSelector(selection_mode="learned")
        sel.train()
        head = SpatialAttentionHead()
        head.train()
        B, N, D = 1, 225, 16
        tokens = torch.randn(B, N, D)
        logits = torch.randn(B, N, requires_grad=True)

        mask, indices = sel(logits)
        h, _ = head(tokens, logits, mask, indices)
        (h ** 2).sum().backward()

        assert logits.grad is not None
        selected = indices[0]
        unselected = ~torch.isin(torch.arange(N), selected)
        assert logits.grad[0, unselected].abs().sum().item() > 0, (
            "Unselected logits must receive STE gradient in learned mode"
        )

    def test_fixed_unselected_have_zero_gradient(self):
        """Fixed mode: unselected scorer logits receive NO gradient
        through the selection path (only through logits in weights)."""
        sel = TopKGumbelSelector(selection_mode="fixed_uniform")
        sel.train()
        head = SpatialAttentionHead()
        head.train()
        B, N, D = 1, 225, 16
        tokens = torch.randn(B, N, D)
        logits = torch.randn(B, N, requires_grad=True)

        mask, indices = sel(logits)
        h, _ = head(tokens, logits, mask, indices)
        (h ** 2).sum().backward()

        assert logits.grad is not None
        selected = indices[0]
        unselected = ~torch.isin(torch.arange(N), selected)
        # In fixed mode, unselected logits only have gradient through
        # the weight normalisation path (exp(logits) * mask).  With mask
        # being K-hot, gradient from unselected positions is zero because
        # mask is zero there and the chain rule gives d(h)/d(logits_u) = 0
        # when the weight contribution from u is zero.
        assert logits.grad[0, unselected].abs().sum().item() == 0.0, (
            "Unselected logits must NOT receive selection gradient in fixed mode"
        )


# ===================================================================
# Model forward integration
# ===================================================================

class TestModelIntegration:
    def _test_forward(self, mode, **kw):
        from rwm.models.rwm.model import ReducedWorldModel
        model = ReducedWorldModel(action_dim=3, selection_mode=mode, **kw)
        model.eval()
        img = torch.randn(1, 3, 64, 64)
        act = torch.zeros(1, 3)
        out = model(img=img, prev_action=act, current_action=act, force_keep_input=True)
        assert out.reward_pred.shape == (1, 1)
        assert out.indices.shape == (1, K)

    def test_learned(self):
        self._test_forward("learned")

    def test_fixed_uniform(self):
        self._test_forward("fixed_uniform")

    def test_fixed_random(self):
        self._test_forward("fixed_random", selection_seed=7)


# ===================================================================
# Checkpoint compatibility
# ===================================================================

class TestCheckpointCompat:
    def test_legacy_checkpoint_no_fixed_indices(self, tmp_path):
        """A checkpoint saved without ``_fixed_indices`` (pre-selection-config
        or with persistent=False) must load as learned/K=8."""
        from rwm.models.rwm.model import ReducedWorldModel
        from rwm.utils.checkpointing import load_checkpoint, model_from_checkpoint
        # Create a state dict that has NO _fixed_indices key
        model = ReducedWorldModel(action_dim=3, selection_mode="learned")
        state = model.state_dict()
        assert "selector._fixed_indices" not in state  # non-persistent
        torch.save(state, tmp_path / "legacy.pt")
        ckpt = load_checkpoint(tmp_path / "legacy.pt")
        loaded = model_from_checkpoint(ckpt, action_dim=3)
        assert loaded._selection_mode == "learned"
        assert loaded._selection_k == 8

    def test_broken_period_checkpoint_has_fixed_indices_dropped(self, tmp_path):
        """A bare state_dict with ``_fixed_indices`` key (from the broken
        register_buffer period) must have that key dropped silently when
        loaded into a fixed-mode model via ``load_state_dict``."""
        from rwm.models.rwm.model import ReducedWorldModel
        # Simulate a bare state dict with the broken key
        ref_model = ReducedWorldModel(action_dim=3, selection_mode="fixed_uniform")
        state = ref_model.state_dict()
        state["selector._fixed_indices"] = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

        # Load into a fresh model — the hook drops _fixed_indices and
        # reconstructs from selection_mode/k.
        model = ReducedWorldModel(action_dim=3, selection_mode="fixed_uniform")
        model.load_state_dict(state, strict=True)
        # Should not have crashed.  Fixed indices should be correct.
        from rwm.models.rwm.topk_gumbel_selector import _farthest_point_sample
        expected = _farthest_point_sample(8)
        assert model.selector._fixed_indices is not None
        torch.testing.assert_close(model.selector._fixed_indices, expected)

    def test_real_stage02_anchor_loads(self):
        """Load the actual Stage-02 linear anchor checkpoint through
        ``model_from_checkpoint()``."""
        from rwm.utils.checkpointing import load_checkpoint, model_from_checkpoint
        from pathlib import Path
        ckpt_path = Path("runs/component_refinement/causal_transformer/02_vectorized_reward_anchor/beta0.1_seed42/checkpoint_best.pt")
        if not ckpt_path.exists():
            pytest.skip("Stage-02 checkpoint not found")
        ckpt = load_checkpoint(ckpt_path)
        model = model_from_checkpoint(ckpt, action_dim=3)
        model.eval()
        assert model._selection_mode == "learned"
        assert model._selection_k == 8


# ===================================================================
# Attention trace
# ===================================================================

class TestTrace:
    def test_trace_reports_mode_and_k(self):
        from rwm.evaluation.attention_trace import trace_attention
        from rwm.models.rwm.model import ReducedWorldModel
        import torch

        for mode in ("learned", "fixed_uniform", "fixed_random"):
            model = ReducedWorldModel(action_dim=3, selection_mode=mode,
                                      selection_seed=42)
            model.eval()
            img = torch.randn(1, 3, 64, 64)
            trace = trace_attention(model, img)
            assert trace.selection_mode == mode
            assert trace.selection_k == 8


# ===================================================================
# Parameterised K=4/8/16/32 tests
# ===================================================================

class TestMultipleK:
    @pytest.mark.parametrize("k", _KS)
    def test_model_forward_shapes(self, k):
        model = ReducedWorldModel(action_dim=3, selection_mode="learned", selection_k=k)
        model.eval()
        img = torch.randn(2, 3, 64, 64)
        act = torch.zeros(2, 3)
        out = model(img=img, prev_action=act, current_action=act, force_keep_input=True)
        assert out.reward_pred.shape == (2, 1)
        assert out.indices.shape == (2, k)

    @pytest.mark.parametrize("k", _KS)
    def test_selector_unique_indices(self, k):
        sel = TopKGumbelSelector(k=k, selection_mode="learned")
        sel.eval()
        logits = torch.randn(4, 225)
        _, indices = sel(logits)
        assert indices.shape == (4, k)
        for b in range(4):
            assert len(indices[b].unique()) == k
            assert (indices[b] >= 0).all() and (indices[b] < 225).all()

    @pytest.mark.parametrize("k", _KS)
    def test_attention_trace_reports_k(self, k):
        model = ReducedWorldModel(action_dim=3, selection_mode="learned", selection_k=k)
        model.eval()
        img = torch.randn(1, 3, 64, 64)
        trace = trace_attention(model, img)
        assert trace.selection_k == k
        assert trace.indices.shape == (1, k)

    @pytest.mark.parametrize("k", _KS)
    def test_overlay_shape(self, k):
        from rwm.evaluation.attention_trace import render_selected_overlay, AttentionTrace
        logits = torch.randn(1, 225)
        indices = torch.randint(0, 225, (1, k))
        weights = torch.randn(1, k).softmax(dim=-1)
        trace = AttentionTrace(logits=logits, indices=indices, weights=weights)
        overlay = render_selected_overlay(trace)
        assert overlay.shape == (1, 64, 64)

    @pytest.mark.parametrize("k", _KS)
    def test_eval_deterministic(self, k):
        sel = TopKGumbelSelector(k=k, selection_mode="learned")
        sel.eval()
        logits = torch.randn(2, 225)
        _, idx1 = sel(logits)
        _, idx2 = sel(logits)
        torch.testing.assert_close(idx1, idx2)

    @pytest.mark.parametrize("k", _KS)
    def test_training_ste_gradient(self, k):
        """For each K, unselected logits must receive STE gradient via
        the spatial-head path."""
        sel = TopKGumbelSelector(k=k, selection_mode="learned")
        sel.train()
        head = SpatialAttentionHead()
        head.train()
        B, N, D = 1, 225, 16
        tokens = torch.randn(B, N, D)
        logits = torch.randn(B, N, requires_grad=True)
        mask, indices = sel(logits)
        h, _ = head(tokens, logits, mask, indices)
        (h ** 2).sum().backward()
        assert logits.grad is not None
        selected = indices[0]
        unselected = ~torch.isin(torch.arange(N), selected)
        assert logits.grad[0, unselected].abs().sum().item() > 0, (
            f"K={k}: unselected logits must receive STE gradient"
        )


# ===================================================================
# Checkpoint K=16 round-trip
# ===================================================================

class TestCheckpointK16:
    def test_structured_restores_k16(self, tmp_path):
        model = ReducedWorldModel(action_dim=3, selection_mode="learned", selection_k=16,
                                  tokenizer_eval_mode="mean")
        model.eval()
        cfg = ExperimentConfig(
            perception=PerceptionConfig(selection_mode="learned", k=16, tokenizer_eval_mode="mean"),
        )
        ckpt_path = save_checkpoint(tmp_path / "k16", model_state=model.state_dict(), config=cfg)
        ckpt = load_checkpoint(ckpt_path)
        loaded = model_from_checkpoint(ckpt, action_dim=3)
        loaded.eval()
        assert loaded._selection_k == 16
        assert loaded.selector.k == 16
        img = torch.randn(1, 3, 64, 64)
        act = torch.zeros(1, 3)
        with torch.no_grad():
            o1 = model(img=img, prev_action=act, current_action=act, force_keep_input=True)
            o2 = loaded(img=img, prev_action=act, current_action=act, force_keep_input=True)
        torch.testing.assert_close(o1.reward_pred, o2.reward_pred)

    def test_legacy_still_restores_k8(self, tmp_path):
        """Legacy checkpoint (no perception config) must restore as learned/K=8."""
        model = ReducedWorldModel(action_dim=3, selection_mode="learned", selection_k=8)
        torch.save(model.state_dict(), tmp_path / "legacy.pt")
        ckpt = load_checkpoint(tmp_path / "legacy.pt")
        loaded = model_from_checkpoint(ckpt, action_dim=3)
        assert loaded._selection_k == 8
        assert loaded._selection_mode == "learned"
