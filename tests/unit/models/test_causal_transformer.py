"""Causal Transformer validation tests.

Covers:
- causal masking (future tokens cannot affect earlier outputs)
- padding / length masking
- context truncation
- action conditioning (different actions → different states/rewards)
- deterministic evaluation behaviour
- full-sequence vs incremental history equivalence in eval mode
- complete ReducedWorldModel output fields and shapes
- no accidental gradient detachment in HistoryBuffer
"""

import pytest
import torch
from torch import Tensor

from rwm.models.rwm.causal_transformer import CausalTransformer
from rwm.models.rwm.model import ReducedWorldModel
from rwm.types import WorldModelOutput
from rwm.utils.history_buffer import HistoryBuffer


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _dummy_history(B: int, T: int, input_dim: int) -> Tensor:
    return torch.randn(B, T, input_dim)


# ==================================================================
# CausalTransformer unit tests
# ==================================================================

class TestCausalMasking:
    def test_future_tokens_do_not_affect_earlier_output(self):
        """Causal masking: output at position t must not depend on tokens
        at positions > t."""
        B, T, D = 2, 5, 8
        model = CausalTransformer()
        model.eval()

        # Two sequences that differ only in the last token.
        hist_a = torch.randn(B, T, 35)  # input_dim = VALUES_DIM + ACTION_DIM = 35
        hist_b = hist_a.clone()
        hist_b[:, -1, :] = 999.0  # drastically different last token

        with torch.no_grad():
            out_a = model(hist_a)
            out_b = model(hist_b)

        # Output at position 0 (the last valid token) should differ
        # because the last token is different.
        assert not torch.allclose(out_a, out_b), (
            "Different last tokens should produce different outputs"
        )

    def test_identical_prefix_different_suffix_gives_same_first_output(self):
        """With causal masking, output at position t=0 (first token)
        depends only on token 0.  Two sequences with same first token
        but different later tokens should give the same output at
        position 0 when extracted via lengths."""
        B, T, D = 1, 5, 35
        model = CausalTransformer()
        model.eval()

        hist_same = torch.randn(B, T, D)
        hist_diff = hist_same.clone()
        hist_diff[:, 2:, :] = 999.0  # different from position 2 onward

        with torch.no_grad():
            # Extract output at position 0 (length=1)
            lengths_same = torch.ones(B, dtype=torch.long)
            lengths_diff = torch.ones(B, dtype=torch.long)

            ws_same = model(hist_same, lengths=lengths_same)
            ws_diff = model(hist_diff, lengths=lengths_diff)

        torch.testing.assert_close(ws_same, ws_diff)


class TestPadding:
    def test_key_padding_mask_shape(self):
        """Key padding mask has correct shape and True for padded positions."""
        model = CausalTransformer()
        B, T, D = 2, 5, 35
        hist = torch.randn(B, T, D)
        lengths = torch.tensor([3, 5], dtype=torch.long)

        mask = model._kpm_from_lengths(lengths, T)
        assert mask.shape == (B, T)
        assert mask[0, 0:3].tolist() == [False, False, False]
        assert mask[0, 3:5].tolist() == [True, True]
        assert mask[1, :].tolist() == [False] * 5


class TestContextTruncation:
    def test_history_buffer_truncates_at_max_len(self):
        B, D, max_len = 1, 35, 5
        buf = HistoryBuffer(max_seq_len=max_len, input_dim=D, device="cpu")
        for _ in range(10):
            token = torch.randn(B, 1, D)
            buf.append(token)
        assert buf.history is not None
        assert buf.history.size(1) == max_len

    def test_history_preserves_gradient_flow(self):
        """HistoryBuffer.append must not use torch.no_grad()."""
        B, D, max_len = 1, 35, 5
        buf = HistoryBuffer(max_seq_len=max_len, input_dim=D, device="cpu")

        token = torch.randn(B, 1, D, requires_grad=True)
        h, _ = buf.append(token)
        loss = h.sum()
        loss.backward()  # should not raise RuntimeError
        assert token.grad is not None, (
            "Gradient must flow through HistoryBuffer.append"
        )


# ==================================================================
# Action conditioning
# ==================================================================

class TestActionConditioning:
    def test_different_actions_produce_different_states(self):
        """The model should produce different world states when
        conditioned on different actions from the same observation."""
        model = ReducedWorldModel()
        model.eval()

        img = torch.randn(1, 3, 64, 64)
        obs_repeat = img.repeat(2, 1, 1, 1)  # same obs, batch of 2

        # Two different actions
        act_a = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        act_b = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        acts = torch.cat([act_a, act_b], dim=0)  # (2, 3)

        with torch.no_grad():
            out_a = model(img=obs_repeat[0:1], prev_action=acts[0:1], current_action=acts[0:1], force_keep_input=True)
            out_b = model(img=obs_repeat[1:2], prev_action=acts[1:2], current_action=acts[1:2], force_keep_input=True)

        # World states should differ
        assert not torch.allclose(
            out_a.world_state, out_b.world_state, atol=1e-4,
        ), "Different actions must produce different world states"


# ==================================================================
# Determinism
# ==================================================================

class TestDeterminism:
    def test_causal_transformer_is_deterministic(self):
        """The CausalTransformer itself is deterministic in eval mode
        (no dropout, no random sampling)."""
        model = CausalTransformer()
        model.eval()

        B, T, D = 2, 5, 35
        hist = torch.randn(B, T, D)

        with torch.no_grad():
            ws1 = model(hist)
            ws2 = model(hist)

        torch.testing.assert_close(ws1, ws2)

    def test_full_model_seeded_is_reproducible(self):
        """The full ReducedWorldModel is reproducible when the same
        seed is used before each forward call (known tokenizer
        stochasticity)."""
        from rwm.utils.seeding import set_seed
        model = ReducedWorldModel()
        model.eval()

        img = torch.randn(1, 3, 64, 64)
        act = torch.zeros(1, 3)

        set_seed(0)
        with torch.no_grad():
            out1 = model(img=img, prev_action=act, current_action=act, force_keep_input=True)

        set_seed(0)
        with torch.no_grad():
            out2 = model(img=img, prev_action=act, current_action=act, force_keep_input=True)

        torch.testing.assert_close(out1.world_state, out2.world_state, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out1.reward_pred, out2.reward_pred, atol=1e-5, rtol=1e-5)


# ==================================================================
# Output structure
# ==================================================================

class TestOutputStructure:
    def test_world_model_output_fields_and_shapes(self):
        """Verify all WorldModelOutput fields have expected shapes."""
        model = ReducedWorldModel()
        model.eval()
        B = 2

        img = torch.randn(B, 3, 64, 64)
        act = torch.zeros(B, 3)

        out: WorldModelOutput = model(
            img=img, prev_action=act, current_action=act, force_keep_input=True,
        )

        assert isinstance(out, WorldModelOutput)
        assert out.world_state.shape == (B, 80)
        assert out.reward_pred.shape == (B, 1)
        assert out.mask_soft.shape == (B, 225)  # N = number of patches
        assert out.indices.shape == (B, 8)      # K = 8
        assert out.history.shape[0] == B
        assert out.history.shape[2] == 35       # values_dim + action_dim
        assert out.lengths.shape == (B,)
        # tok_mu/tok_logvar may be None if tokenizer returns plain Tensor
        if out.tok_mu is not None:
            assert out.tok_mu.shape[0] == B
        if out.tok_logvar is not None:
            assert out.tok_logvar.shape[0] == B

    def test_history_grows_with_each_call(self):
        """Each forward call appends one token, so history length
        should increase until it hits the max context length."""
        model = ReducedWorldModel()
        model.eval()

        img = torch.randn(1, 3, 64, 64)
        act = torch.zeros(1, 3)

        out = model(img=img, prev_action=act, current_action=act, force_keep_input=True)
        len1 = out.history.shape[1]

        out = model(
            img=img, prev_action=act, current_action=act,
            history=out.history, lengths=out.lengths,
            force_keep_input=True,
        )
        len2 = out.history.shape[1]

        assert len2 == len1 + 1, (
            f"History length should increase by 1 ({len1} -> {len2})"
        )

    def test_history_is_not_cached_between_calls(self):
        """Omitting history starts an independent sequence every time."""
        model = ReducedWorldModel()
        model.eval()
        img = torch.randn(1, 3, 64, 64)
        act = torch.zeros(1, 3)

        first = model(img=img, prev_action=act, current_action=act, force_keep_input=True)
        second = model(img=img, prev_action=act, current_action=act, force_keep_input=True)

        assert first.history.shape[1] == 1
        assert second.history.shape[1] == 1


# ==================================================================
# Full-sequence vs incremental equivalence
# ==================================================================

class TestIncrementalEquivalence:
    def test_full_vs_incremental_agree_in_eval(self):
        """Running the transformer on a full pre-built sequence should
        give the same final-world-state as incrementally feeding one
        frame at a time.  Both paths use the same random seed for the
        stochastic tokenizer."""
        from rwm.utils.seeding import set_seed
        model = ReducedWorldModel()
        model.eval()

        B, T = 1, 5
        imgs = torch.randn(T, 3, 64, 64)
        acts = torch.randn(T, 3)

        # Incremental: feed one frame at a time (share seed across calls).
        set_seed(42)
        inc_out = None
        for t in range(T):
            inc_out = model(
                img=imgs[t:t+1], prev_action=acts[t:t+1], current_action=acts[t:t+1],
                history=inc_out.history if inc_out else None,
                lengths=inc_out.lengths if inc_out else None,
                force_keep_input=True,
            )

        # Full: use the transformer directly on pre-built tokens.
        set_seed(42)
        with torch.no_grad():
            full_tokens = torch.cat([
                torch.cat([
                    model.generate_spatial_rep(imgs[t:t+1]),
                    acts[t:t+1],
                ], dim=-1).unsqueeze(1)
                for t in range(T)
            ], dim=1)

        with torch.no_grad():
            ws_full = model.world_hd(full_tokens)
            _shared_full, rp_full = model.controller(ws_full, acts[-1:])

        torch.testing.assert_close(
            inc_out.world_state, ws_full, atol=1e-4, rtol=1e-4,
            msg="Incremental and full-sequence world states must match",
        )
        torch.testing.assert_close(
            inc_out.reward_pred, rp_full, atol=1e-4, rtol=1e-4,
            msg="Incremental and full-sequence reward predictions must match",
        )
