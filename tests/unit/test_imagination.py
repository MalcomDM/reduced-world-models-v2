"""Tests for the differentiable bounded-context imagination interface (Stage 3.0).

Verifies:
  1. Observed warmup matches ``forward_sequence`` / incremental visible outputs.
  2. One masked factual advance matches the D.0 masked interface.
  3. Multi-step bounded rollout has correct history truncation and tensor shapes.
  4. Score-then-advance action timing is proved with a spy.
  5. Gradients from summed imagined rewards reach input action tensors and
     world-model parameters.
  6. No future observation can affect imagined states/rewards.
  7. Existing visible reward APIs remain unchanged.
"""

from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from rwm.config.config import ACTION_DIM, VALUES_DIM, SEQ_LEN
from rwm.imagination import ImaginationRollout, RolloutOutput
from rwm.models.rwm.model import ReducedWorldModel
from rwm.types import WorldModelOutput


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
    return m


@pytest.fixture
def imag(model):
    return ImaginationRollout(model)


@pytest.fixture
def syn_batch():
    """Synthetic observed batch: B=2, T=6, timing contract exact.

    Timing::
        prev_actions[:, t] = action[t-1]   (zeros at t=0)
        current_actions[:, t] = action[t]
    """
    B, T, C, H, W = 2, 6, 3, 64, 64
    obs = torch.randn(B, T, C, H, W)
    current_actions = torch.randn(B, T, ACTION_DIM)
    prev_actions = torch.zeros(B, T, ACTION_DIM)
    if T > 1:
        prev_actions[:, 1:] = current_actions[:, :-1]
    return dict(obs=obs, prev_actions=prev_actions,
                current_actions=current_actions)


# ---------------------------------------------------------------------------
# Test 1: Warmup matches forward_sequence
# ---------------------------------------------------------------------------

class TestWarmupMatchesForwardSequence:
    """Observed warmup through the new interface must produce the same
    history tokens and per-step beliefs as ``forward_sequence``."""

    def test_history_matches(self, imag, syn_batch):
        state = imag.warmup(**syn_batch, force_keep_input=True)
        out = imag.model.forward_sequence(
            syn_batch["obs"], syn_batch["prev_actions"],
            syn_batch["current_actions"], force_keep_input=True,
        )
        torch.testing.assert_close(state.history, out.history)

    def test_beliefs_match_world_hd(self, imag, syn_batch):
        state = imag.warmup(**syn_batch, force_keep_input=True)
        out = imag.model.forward_sequence(
            syn_batch["obs"], syn_batch["prev_actions"],
            syn_batch["current_actions"], force_keep_input=True,
        )
        expected_beliefs = imag.model.world_hd(
            out.history, lengths=out.lengths, return_all=True,
        )
        torch.testing.assert_close(state.beliefs, expected_beliefs)

    def test_current_belief_is_last_warmup_step(self, imag, syn_batch):
        state = imag.warmup(**syn_batch, force_keep_input=True)
        torch.testing.assert_close(
            state.current_belief, state.beliefs[:, -1, :],
        )

    def test_warmup_without_current_actions(self, imag, syn_batch):
        """warmup should succeed when current_actions is None."""
        state = imag.warmup(
            syn_batch["obs"], syn_batch["prev_actions"],
            current_actions=None, force_keep_input=True,
        )
        T = syn_batch["obs"].shape[1]
        assert state.history.shape[1] == T


# ---------------------------------------------------------------------------
# Test 2: Warmup matches incremental forward
# ---------------------------------------------------------------------------

class TestWarmupMatchesIncremental:
    """Step-by-step ``model.forward()`` must produce the same history
    as a single warmup call."""

    def test_incremental_vs_warmup(self, imag, syn_batch):
        B, T = syn_batch["obs"].shape[0], syn_batch["obs"].shape[1]

        # Incremental forward.
        buf = None
        lengths = None
        for t in range(T):
            prev_a = syn_batch["prev_actions"][:, t, :]
            curr_a = syn_batch["current_actions"][:, t, :]
            img_t = syn_batch["obs"][:, t, :, :, :]
            out = imag.model(
                img=img_t, prev_action=prev_a, current_action=curr_a,
                history=buf, lengths=lengths, force_keep_input=True,
            )
            buf = out.history
            lengths = out.lengths

        incremental_history = buf

        # Warmup.
        state = imag.warmup(**syn_batch, force_keep_input=True)

        torch.testing.assert_close(state.history, incremental_history)


# ---------------------------------------------------------------------------
# Test 3: Advance matches D.0 masked interface
# ---------------------------------------------------------------------------

class TestAdvanceMatchesMasked:
    """One blind advance must produce the same tokens and belief as ``forward_sequence`` with``observation_keep`` masking."""

    def _masked_sequence(self, imag, syn_batch, warmup_steps=4, mask_horizon=4):
        """Build a sequence with a masked block and return the D.0-style
        ``forward_sequence`` output for the *first* masked position."""
        B, T_full = syn_batch["obs"].shape[0], syn_batch["obs"].shape[1]

        # Build observation_keep: visible for warmup, masked for horizon.
        keep = torch.ones(B, T_full, dtype=torch.bool, device=syn_batch["obs"].device)
        mask_end = min(warmup_steps + mask_horizon, T_full)
        if mask_end > warmup_steps:
            keep[:, warmup_steps:mask_end] = False

        out = imag.model.forward_sequence(
            syn_batch["obs"], syn_batch["prev_actions"],
            syn_batch["current_actions"],
            force_keep_input=True,
            observation_keep=keep,
        )

        # The first masked position is at index warmup_steps.
        t_mask = warmup_steps
        masked_belief = imag.model.world_hd(out.history, lengths=out.lengths, return_all=True)[:, t_mask, :]
        masked_reward = out.reward_pred_seq[:, t_mask]

        return out.history, out.lengths, masked_belief, masked_reward

    def test_advance_belief_matches_masked(self, imag, syn_batch):
        """Advance one step from the warmup prefix must give the same
        belief as the first masked position in the full masked sequence.

        The advance appends ``cat(zeros, action)`` where *action* is the
        previous action (``action[warmup_steps-1]``), matching the token at
        the masked position ``t = warmup_steps``.
        """
        warmup_steps = 4

        # Warmup from first warmup_steps steps only.
        warmup_obs = syn_batch["obs"][:, :warmup_steps, :, :, :]
        warmup_prev = syn_batch["prev_actions"][:, :warmup_steps, :]
        warmup_curr = syn_batch["current_actions"][:, :warmup_steps, :]

        state = imag.warmup(warmup_obs, warmup_prev, warmup_curr, force_keep_input=True)

        # The blind token at the first masked position carries
        # ``prev_action`` = ``action[warmup_steps-1]``
        # (= ``prev_actions[:, warmup_steps, :]`` in the D.0 protocol).
        prev_at_mask_start = syn_batch["prev_actions"][:, warmup_steps, :]

        _, _, new_belief = imag.advance(
            state.history, state.lengths, prev_at_mask_start,
        )

        # Ground truth: masked belief from full-sequence forward_sequence.
        _, _, masked_belief, _ = self._masked_sequence(
            imag, syn_batch, warmup_steps=warmup_steps, mask_horizon=4,
        )

        torch.testing.assert_close(new_belief, masked_belief, atol=1e-5, rtol=1e-5)

    def test_advance_history_length_correct(self, imag, syn_batch):
        warmup_steps = 4
        warmup_obs = syn_batch["obs"][:, :warmup_steps, :, :, :]
        warmup_prev = syn_batch["prev_actions"][:, :warmup_steps, :]
        warmup_curr = syn_batch["current_actions"][:, :warmup_steps, :]

        state = imag.warmup(warmup_obs, warmup_prev, warmup_curr, force_keep_input=True)
        advance_action = syn_batch["prev_actions"][:, warmup_steps, :]

        new_hist, new_lens, _ = imag.advance(
            state.history, state.lengths, advance_action,
        )

        assert new_hist.shape[1] == warmup_steps + 1
        assert (new_lens == warmup_steps + 1).all()
        assert new_hist.device == state.history.device


# ---------------------------------------------------------------------------
# Test 4: Multi-step rollout — shapes and truncation
# ---------------------------------------------------------------------------

class TestRolloutShapes:
    def test_rollout_shapes(self, imag, syn_batch):
        B, T_warm = syn_batch["obs"].shape[0], 4
        H = 3

        warmup_obs = syn_batch["obs"][:, :T_warm, :, :, :]
        warmup_prev = syn_batch["prev_actions"][:, :T_warm, :]
        warmup_curr = syn_batch["current_actions"][:, :T_warm, :]

        state = imag.warmup(warmup_obs, warmup_prev, warmup_curr, force_keep_input=True)

        actions = torch.randn(B, H, ACTION_DIM, device=syn_batch["obs"].device)
        out = imag.rollout(state.history, state.lengths, state.current_belief, actions)

        assert isinstance(out, RolloutOutput)
        B_, D = state.current_belief.shape
        assert out.states.shape == (B, H, D)
        assert out.actions.shape == (B, H, ACTION_DIM)
        assert out.rewards.shape == (B, H)
        assert out.next_state.shape == (B, D)
        assert out.history.shape[1] == T_warm + H  # no truncation yet

    def test_rollout_truncation(self, imag, syn_batch):
        """When warmup + rollout exceeds SEQ_LEN, history must be truncated."""
        B = syn_batch["obs"].shape[0]
        T_warm = 18
        H = 5  # T_warm + H = 23 > SEQ_LEN (20)

        # Extend batch to T_warm steps.
        device = syn_batch["obs"].device
        big_obs = torch.randn(B, T_warm, 3, 64, 64, device=device)
        big_prev = torch.randn(B, T_warm, ACTION_DIM, device=device)
        big_curr = torch.randn(B, T_warm, ACTION_DIM, device=device)

        state = imag.warmup(big_obs, big_prev, big_curr, force_keep_input=True)
        assert state.history.shape[1] <= SEQ_LEN
        assert state.history.shape[1] == T_warm  # within SEQ_LEN

        actions = torch.randn(B, H, ACTION_DIM, device=device)
        out = imag.rollout(state.history, state.lengths, state.current_belief, actions)

        # History must be truncated to SEQ_LEN.
        assert out.history.shape[1] == SEQ_LEN, (
            f"Expected truncation to {SEQ_LEN}, got {out.history.shape[1]}"
        )
        assert (out.lengths == SEQ_LEN).all()

    def test_rollout_preserves_warmup_prefix(self, imag, syn_batch):
        """The warmup prefix must appear unchanged at the start of the
        rollout history."""
        B, T_warm = syn_batch["obs"].shape[0], 4
        H = 3

        warmup_obs = syn_batch["obs"][:, :T_warm, :, :, :]
        warmup_prev = syn_batch["prev_actions"][:, :T_warm, :]
        warmup_curr = syn_batch["current_actions"][:, :T_warm, :]

        state = imag.warmup(warmup_obs, warmup_prev, warmup_curr, force_keep_input=True)
        warmup_history = state.history.clone()

        actions = torch.randn(B, H, ACTION_DIM, device=syn_batch["obs"].device)
        out = imag.rollout(state.history, state.lengths, state.current_belief, actions)

        # The first T_warm tokens in the rollout history should match the
        # original warmup history.
        torch.testing.assert_close(out.history[:, :T_warm, :], warmup_history)

    def test_rollout_score_order(self, imag, syn_batch):
        """The first reward is scored from the initial belief, not after
        advancing.  Verify by checking that the first reward matches
        score(initial_belief, first_action)."""
        B, T_warm = syn_batch["obs"].shape[0], 4
        H = 1

        warmup_obs = syn_batch["obs"][:, :T_warm, :, :, :]
        warmup_prev = syn_batch["prev_actions"][:, :T_warm, :]
        warmup_curr = syn_batch["current_actions"][:, :T_warm, :]

        state = imag.warmup(warmup_obs, warmup_prev, warmup_curr, force_keep_input=True)
        action = torch.randn(B, H, ACTION_DIM, device=syn_batch["obs"].device)

        out = imag.rollout(state.history, state.lengths, state.current_belief, action)

        expected_reward = imag.score(state.current_belief, action[:, 0, :])
        torch.testing.assert_close(out.rewards[:, 0].unsqueeze(-1), expected_reward)


# ---------------------------------------------------------------------------
# Test 5: Score-then-advance timing (spy)
# ---------------------------------------------------------------------------

class _ControllerSpy(nn.Module):
    """Records what is passed to ``ControllerTrunk.encode`` and
    ``predict_reward`` during a rollout.

    Must be an ``nn.Module`` subclass so that ``model.controller``
    can be replaced (PyTorch requires child modules to be ``nn.Module``).
    """

    def __init__(self, real):
        super().__init__()
        self._real = real
        self.encode_inputs: list[torch.Tensor] = []
        self.reward_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def encode(self, belief: torch.Tensor) -> torch.Tensor:
        self.encode_inputs.append(belief.detach().cpu())
        return self._real.encode(belief)

    def predict_reward(
        self, shared_repr: torch.Tensor, action: torch.Tensor,
    ) -> torch.Tensor:
        self.reward_inputs.append((
            shared_repr.detach().cpu(),
            action.detach().cpu(),
        ))
        return self._real.predict_reward(shared_repr, action)

    def forward(self, belief, action):
        """Delegate to the real controller's forward."""
        h = self.encode(belief)
        r = self.predict_reward(h, action)
        return h, r


class TestScoreThenAdvanceTiming:
    """Score is called with the current belief and *current* action;
    advance then appends that same action as the *previous* action for
    the next step."""

    @pytest.mark.models
    def test_first_step_uses_initial_belief(self, imag, syn_batch):
        B, T_warm = syn_batch["obs"].shape[0], 4
        H = 2

        warmup_obs = syn_batch["obs"][:, :T_warm, :, :, :]
        warmup_prev = syn_batch["prev_actions"][:, :T_warm, :]
        warmup_curr = syn_batch["current_actions"][:, :T_warm, :]

        state = imag.warmup(warmup_obs, warmup_prev, warmup_curr, force_keep_input=True)
        actions = torch.randn(B, H, ACTION_DIM, device=syn_batch["obs"].device)

        spy = _ControllerSpy(imag.model.controller)
        imag.model.controller = spy

        _ = imag.rollout(state.history, state.lengths, state.current_belief, actions)

        assert len(spy.encode_inputs) == H
        assert len(spy.reward_inputs) == H

        # First encode receives initial_belief.
        init_belief = state.current_belief.detach().cpu()
        torch.testing.assert_close(spy.encode_inputs[0], init_belief)

        # First reward head receives (encode(initial_belief), actions[:,0]).
        encoded_0 = spy._real.encode(state.current_belief).detach().cpu()
        act_0 = actions[:, 0, :].detach().cpu()
        torch.testing.assert_close(spy.reward_inputs[0][0], encoded_0)
        torch.testing.assert_close(spy.reward_inputs[0][1], act_0)

    @pytest.mark.models
    def test_second_step_belief_changed(self, imag, syn_batch):
        """After one advance, the belief used for scoring must differ from
        the initial belief."""
        B, T_warm = syn_batch["obs"].shape[0], 4
        H = 2

        warmup_obs = syn_batch["obs"][:, :T_warm, :, :, :]
        warmup_prev = syn_batch["prev_actions"][:, :T_warm, :]
        warmup_curr = syn_batch["current_actions"][:, :T_warm, :]

        state = imag.warmup(warmup_obs, warmup_prev, warmup_curr, force_keep_input=True)
        actions = torch.randn(B, H, ACTION_DIM, device=syn_batch["obs"].device)

        spy = _ControllerSpy(imag.model.controller)
        imag.model.controller = spy

        _ = imag.rollout(state.history, state.lengths, state.current_belief, actions)

        # Belief at step 1 must differ from step 0 (because advance
        # appended a new token).
        assert not torch.allclose(spy.encode_inputs[0], spy.encode_inputs[1])


# ---------------------------------------------------------------------------
# Test 6: Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    @pytest.mark.models
    def test_gradients_reach_action_tensor(self, imag, syn_batch):
        """Gradients from summed imagined rewards must flow to the input
        action tensor."""
        imag.train()
        imag.model.train()

        B, T_warm = syn_batch["obs"].shape[0], 4
        H = 3

        warmup_obs = syn_batch["obs"][:, :T_warm, :, :, :]
        warmup_prev = syn_batch["prev_actions"][:, :T_warm, :]
        warmup_curr = syn_batch["current_actions"][:, :T_warm, :]

        state = imag.warmup(warmup_obs, warmup_prev, warmup_curr, force_keep_input=True)

        actions = torch.randn(B, H, ACTION_DIM, device=syn_batch["obs"].device,
                              requires_grad=True)
        out = imag.rollout(state.history, state.lengths, state.current_belief, actions)

        loss = out.rewards.sum()
        loss.backward()

        assert actions.grad is not None
        assert actions.grad.abs().sum().item() > 0

    @pytest.mark.models
    def test_gradients_reach_world_model_params(self, imag, syn_batch):
        """Gradients from summed imagined rewards must flow to the world
        model's transformer and controller parameters."""
        imag.train()
        imag.model.train()

        B, T_warm = syn_batch["obs"].shape[0], 4
        H = 3

        warmup_obs = syn_batch["obs"][:, :T_warm, :, :, :]
        warmup_prev = syn_batch["prev_actions"][:, :T_warm, :]
        warmup_curr = syn_batch["current_actions"][:, :T_warm, :]

        state = imag.warmup(warmup_obs, warmup_prev, warmup_curr, force_keep_input=True)
        actions = torch.randn(B, H, ACTION_DIM, device=syn_batch["obs"].device,
                              requires_grad=True)
        out = imag.rollout(state.history, state.lengths, state.current_belief, actions)

        loss = out.rewards.sum()
        loss.backward()

        # Check transformer has gradients.
        trans_has_grad = False
        for p in imag.model.world_hd.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 1e-8:
                trans_has_grad = True
                break
        assert trans_has_grad, "Transformer must receive gradients"

        # Check controller has gradients.
        ctrl_has_grad = False
        for p in imag.model.controller.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 1e-8:
                ctrl_has_grad = True
                break
        assert ctrl_has_grad, "Controller must receive gradients"

    @pytest.mark.models
    def test_warmup_history_does_not_block_gradients(self, imag, syn_batch):
        """The warmup history must not be detached — gradients from the
        rollout must be able to flow back through the warmup tokens into
        the perception stack if the caller requires it."""
        imag.train()
        imag.model.train()

        B, T_warm = syn_batch["obs"].shape[0], 4
        H = 2

        warmup_obs = syn_batch["obs"][:, :T_warm, :, :, :]
        warmup_prev = syn_batch["prev_actions"][:, :T_warm, :]
        warmup_curr = syn_batch["current_actions"][:, :T_warm, :]

        state = imag.warmup(warmup_obs, warmup_prev, warmup_curr, force_keep_input=True)

        actions = torch.randn(B, H, ACTION_DIM, device=syn_batch["obs"].device,
                              requires_grad=True)
        out = imag.rollout(state.history, state.lengths, state.current_belief, actions)

        loss = out.rewards.sum()
        loss.backward()

        # The warmup history tensor should have a grad_fn (it was involved
        # in the computation graph).
        assert state.history.grad_fn is not None


# ---------------------------------------------------------------------------
# Test 7: No future observation affects imagination
# ---------------------------------------------------------------------------

class TestNoFutureObservation:
    """Imagined states and rewards must depend only on the warmup history
    and the imagined actions — never on future observations."""

    def test_imagined_tokens_use_zero_spatial(self, imag, syn_batch):
        """Every imagined token must have a zero spatial representation."""
        B, T_warm = syn_batch["obs"].shape[0], 4
        H = 3

        warmup_obs = syn_batch["obs"][:, :T_warm, :, :, :]
        warmup_prev = syn_batch["prev_actions"][:, :T_warm, :]
        warmup_curr = syn_batch["current_actions"][:, :T_warm, :]

        state = imag.warmup(warmup_obs, warmup_prev, warmup_curr, force_keep_input=True)
        actions = torch.randn(B, H, ACTION_DIM, device=syn_batch["obs"].device)

        out = imag.rollout(state.history, state.lengths, state.current_belief, actions)

        # The spatial part (first VALUES_DIM entries) of each imagined
        # token must be zero.
        imagined_tokens = out.history[:, T_warm:, :VALUES_DIM]
        assert (imagined_tokens == 0).all(), (
            "Imagined steps must have zero spatial representation"
        )

    def test_imagined_rewards_deterministic_given_actions(self, imag, syn_batch):
        """Running rollout twice with the same actions and warmup must
        produce identical rewards."""
        B, T_warm = syn_batch["obs"].shape[0], 4
        H = 3

        warmup_obs = syn_batch["obs"][:, :T_warm, :, :, :]
        warmup_prev = syn_batch["prev_actions"][:, :T_warm, :]
        warmup_curr = syn_batch["current_actions"][:, :T_warm, :]

        state = imag.warmup(warmup_obs, warmup_prev, warmup_curr, force_keep_input=True)
        actions = torch.randn(B, H, ACTION_DIM, device=syn_batch["obs"].device)

        out1 = imag.rollout(state.history, state.lengths, state.current_belief, actions)
        out2 = imag.rollout(state.history, state.lengths, state.current_belief, actions)

        torch.testing.assert_close(out1.rewards, out2.rewards)
        torch.testing.assert_close(out1.states, out2.states)


# ---------------------------------------------------------------------------
# Test 8: Existing visible reward APIs unchanged
# ---------------------------------------------------------------------------

class TestExistingAPIs:
    """The new imagination interface must not modify or break any existing
    visible reward prediction APIs."""

    def test_forward_sequence_still_works(self, model, syn_batch):
        model.eval()
        out = model.forward_sequence(
            syn_batch["obs"], syn_batch["prev_actions"],
            syn_batch["current_actions"], force_keep_input=True,
        )
        assert out.reward_pred_seq.shape == (syn_batch["obs"].shape[0],
                                              syn_batch["obs"].shape[1])

    def test_incremental_forward_still_works(self, model, syn_batch):
        model.eval()
        out = model(
            img=syn_batch["obs"][:, 0, :, :, :],
            prev_action=syn_batch["prev_actions"][:, 0, :],
            current_action=syn_batch["current_actions"][:, 0, :],
            force_keep_input=True,
        )
        assert out.reward_pred.shape == (syn_batch["obs"].shape[0], 1)

    def test_warmup_does_not_alter_model_state(self, imag, syn_batch):
        """Calling warmup must not change the model's eval/train mode."""
        imag.model.eval()
        is_eval_before = not imag.model.training
        _ = imag.warmup(**syn_batch, force_keep_input=True)
        is_eval_after = not imag.model.training
        assert is_eval_before == is_eval_after
