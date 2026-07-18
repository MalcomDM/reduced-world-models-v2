"""Tests for Stage 2.5B.2 — Connected Top-K straight-through gradients.

Verifies:
- Eval forward parity (new method matches old hard gather).
- Hard sparsity (non-selected tokens do not affect eval output).
- Dense training gradient (unselected scorer logits receive gradient).
- Determinism (eval) and variation (train).
- Model integration (forward, forward_sequence, attention trace).
"""

import pytest
import torch
import torch.nn.functional as F

from rwm.models.rwm.topk_gumbel_selector import TopKGumbelSelector
from rwm.models.rwm.spatial_attention_head import SpatialAttentionHead
from rwm.models.rwm.model import ReducedWorldModel
from rwm.evaluation.attention_trace import trace_attention
from rwm.config.config import K, TOKEN_DIM, VALUES_DIM


# ===================================================================
# Eval forward parity
# ===================================================================

class TestEvalParity:
    def test_eval_forward_matches_old_hard_gather(self):
        """SpatialAttentionHead in eval mode must produce the same output
        as the legacy hard gather + selected-logit softmax."""
        torch.manual_seed(0)
        B, N, D = 4, 225, TOKEN_DIM
        tokens = torch.randn(B, N, D)
        logits = torch.randn(B, N)

        selector = TopKGumbelSelector()
        selector.eval()
        head = SpatialAttentionHead()
        head.eval()

        # New path
        selection_mask, indices = selector(logits)
        h_new, attn_new = head(tokens, logits, selection_mask, indices)

        # Legacy hard-gather path
        idx_exp = indices.unsqueeze(-1).expand(-1, -1, D)
        tok_k = tokens.gather(1, idx_exp)
        log_k = logits.gather(1, indices)
        attn_old = F.softmax(log_k / 1.0, dim=1)
        V_k = head.W_v(tok_k)
        h_old = torch.bmm(attn_old.unsqueeze(1), V_k).squeeze(1)
        h_old = head.norm(h_old)

        torch.testing.assert_close(h_new, h_old, atol=1e-5, rtol=1e-5,
                                   msg="Eval output must match legacy hard gather")
        torch.testing.assert_close(attn_new, attn_old, atol=1e-5, rtol=1e-5,
                                   msg="Eval attentions must match legacy softmax")

    def test_eval_projects_only_selected_values(self):
        """The hard inference path must retain the K-token value-projection cost."""
        B, N, D, K = 2, 225, TOKEN_DIM, 8
        tokens = torch.randn(B, N, D)
        logits = torch.randn(B, N)
        selector = TopKGumbelSelector(k=K).eval()
        head = SpatialAttentionHead().eval()
        seen_shapes = []

        hook = head.W_v.register_forward_pre_hook(
            lambda _module, args: seen_shapes.append(tuple(args[0].shape))
        )
        try:
            selection_mask, indices = selector(logits)
            head(tokens, logits, selection_mask, indices)
        finally:
            hook.remove()

        assert seen_shapes == [(B, K, D)]


# ===================================================================
# Hard sparsity
# ===================================================================

class TestHardSparsity:
    def test_non_selected_do_not_affect_eval_output(self):
        """In eval mode, changing non-selected token values must not
        change the spatial head output when selection is fixed."""
        torch.manual_seed(0)
        B, N, D = 1, 225, TOKEN_DIM
        tokens = torch.randn(B, N, D)
        logits = torch.randn(B, N)

        selector = TopKGumbelSelector()
        selector.eval()
        head = SpatialAttentionHead()
        head.eval()

        selection_mask, indices = selector(logits)
        h_ref, _ = head(tokens, logits, selection_mask, indices)

        # Change non-selected tokens drastically
        tokens2 = tokens.clone()
        tokens2[0, ~torch.isin(torch.arange(N), indices[0])] = 999.0
        h2, _ = head(tokens2, logits, selection_mask, indices)

        torch.testing.assert_close(h_ref, h2, atol=1e-5, rtol=1e-5,
                                   msg="Changing non-selected tokens must not affect eval output")


# ===================================================================
# Dense training gradient
# ===================================================================

class TestTrainingGradient:
    def test_unselected_logits_receive_gradient(self):
        """With the STE mask, unselected scorer logit positions must
        receive finite nonzero gradient during training.

        We use ``(h ** 2).sum()`` as the loss because ``h.sum()`` has
        near-zero gradient: the pooling weights sum to 1, so redistributing
        weight among patches has negligible effect when all patches have
        similar values from random initialization.
        """
        torch.manual_seed(0)
        B, N, D = 1, 225, TOKEN_DIM

        selector = TopKGumbelSelector()
        selector.train()
        head = SpatialAttentionHead()
        head.train()

        # Use very different token values so that weight redistribution
        # has a measurable effect on the pooled output.
        tokens = torch.randn(B, N, D) * 10.0
        logits = torch.randn(B, N, requires_grad=True)

        selection_mask, indices = selector(logits)
        h, _ = head(tokens, logits, selection_mask, indices)
        loss = (h ** 2).sum()
        loss.backward()

        assert logits.grad is not None, "logits must receive gradient"
        assert not torch.isnan(logits.grad).any(), "logits gradient must not be NaN"
        assert logits.grad.abs().sum().item() > 1e-8, (
            f"logits gradient must be clearly nonzero; got {logits.grad.abs().sum().item()}"
        )

        # At least one unselected position must have nonzero gradient
        selected_mask = torch.zeros(N, dtype=torch.bool)
        selected_mask[indices[0]] = True
        unselected_grad = logits.grad[0, ~selected_mask]
        assert unselected_grad.abs().sum().item() > 1e-8, (
            "Unselected logit positions must receive gradient"
        )

        # Selected positions must also have gradient
        selected_grad = logits.grad[0, selected_mask]
        assert selected_grad.abs().sum().item() > 1e-8, (
            "Selected logit positions must receive gradient"
        )


# ===================================================================
# Determinism and training variation
# ===================================================================

class TestDeterminism:
    def test_eval_deterministic(self):
        """Eval mode must produce identical indices and output."""
        selector = TopKGumbelSelector()
        selector.eval()
        logits = torch.randn(4, 225)

        _, idx1 = selector(logits)
        _, idx2 = selector(logits)
        torch.testing.assert_close(idx1, idx2)

    def test_train_variation(self):
        """Training mode must produce variation (Gumbel noise)."""
        selector = TopKGumbelSelector()
        selector.train()
        logits = torch.randn(4, 225)

        _, idx1 = selector(logits)
        _, idx2 = selector(logits)
        # Lower bound: at least some samples should differ.
        # With Gumbel noise on 225 dims, Top-8 will differ almost always.
        assert not torch.allclose(idx1, idx2), (
            "Training indices should differ due to Gumbel noise"
        )


# ===================================================================
# Model integration
# ===================================================================

class TestModelIntegration:
    def test_forward_returns_expected_shapes(self):
        """ReducedWorldModel forward must still return expected shapes."""
        model = ReducedWorldModel()
        model.eval()

        B = 2
        img = torch.randn(B, 3, 64, 64)
        act = torch.zeros(B, 3)

        out = model(img=img, prev_action=act, current_action=act, force_keep_input=True)
        assert out.world_state.shape == (B, 80)
        assert out.reward_pred.shape == (B, 1)
        assert out.mask_soft.shape == (B, 225)
        assert out.indices.shape == (B, 8)

    def test_forward_sequence_shapes(self):
        """forward_sequence must return expected shapes."""
        model = ReducedWorldModel()
        model.eval()
        B, T = 2, 5
        obs = torch.randn(B, T, 3, 64, 64)
        prev_acts = torch.zeros(B, T, 3)
        curr_acts = torch.zeros(B, T, 3)

        out = model.forward_sequence(obs, prev_acts, curr_acts, force_keep_input=True)
        assert out.reward_pred_seq.shape == (B, T)
        torch.testing.assert_close(out.reward_pred, out.reward_pred_seq[:, -1:])
        assert out.mask_soft.shape == (B, 225)
        assert out.indices.shape == (B, 8)

    def test_attention_trace_runs_and_preserves_model(self):
        """Attention trace runs and does not change model outputs."""
        model = ReducedWorldModel(tokenizer_eval_mode="mean")
        model.eval()

        img = torch.randn(1, 3, 64, 64)
        act = torch.zeros(1, 3)

        with torch.no_grad():
            out1 = model(img=img, prev_action=act, current_action=act, force_keep_input=True)

        trace = trace_attention(model, img)
        assert trace.logits.shape == (1, 225)
        assert trace.indices.shape == (1, 8)
        assert trace.weights.shape == (1, 8)

        with torch.no_grad():
            out2 = model(img=img, prev_action=act, current_action=act, force_keep_input=True)

        torch.testing.assert_close(out1.world_state, out2.world_state,
                                   msg="Trace must not alter model output")
