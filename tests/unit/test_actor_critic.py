"""Tests for Stage 4.0 Actor-Critic head calibration (termination contract).

Verifies:
  1-3. Action bounds, deterministic mode, log-probability.
  4. λ-returns with the termination/continuation contract.
  5. New contract: terminated, truncated, imagined-horizon semantics.
  6. Critic overfits synthetic returns.
  7. Advantage direction affects log-prob.
  8. Entropy coefficient direction.
  9. Target Critic Polyak update.
 10. Freeze contract: world-model params unchanged.
 11. Validation: terminated+continuation conflict raises.
 12. BoundedGaussian.entropy_sample has finite gradients.
 13. Config round-trip.
"""

import dataclasses

import pytest
import torch

from rwm.config.config import ACTION_DIM, WORLD_STATE_DIM
from rwm.distributions import BoundedGaussian
from rwm.models.actor_critic import (
    Actor,
    ActorCritic,
    ActorCriticConfig,
    Critic,
    compute_lambda_returns,
    compute_td_advantage,
    _validate_tc,
)
from rwm.models.rwm.model import ReducedWorldModel


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _tc(T: int, device="cpu") -> dict:
    """Return default terminated=False, continuation=True, bootstrap_value=0."""
    return dict(
        terminated=torch.zeros(1, T, dtype=torch.bool, device=device),
        continuation=torch.ones(1, T, dtype=torch.bool, device=device),
        bootstrap_value=torch.zeros(1, device=device),
    )


# ---------------------------------------------------------------------------
# Model fixture  shared
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
def ac(model):
    return ActorCritic(model, ActorCriticConfig(hidden_dim=64))


# ===================================================================
# 1. Action bounds
# ===================================================================

class TestActionBounds:
    def test_rsample_respects_bounds(self, ac):
        dist = ac.actor(torch.randn(100, WORLD_STATE_DIM))
        a = dist.rsample()
        assert (a[:, 0] >= -1.0).all() and (a[:, 0] <= 1.0).all()
        assert (a[:, 1:] >= 0.0).all() and (a[:, 1:] <= 1.0).all()

    def test_mode_respects_bounds(self, ac):
        dist = ac.actor(torch.randn(100, WORLD_STATE_DIM))
        a = dist.mode()
        assert (a[:, 0] >= -1.0).all() and (a[:, 0] <= 1.0).all()
        assert (a[:, 1:] >= 0.0).all() and (a[:, 1:] <= 1.0).all()

    def test_rsample_different_each_call(self, ac):
        d = ac.actor(torch.randn(10, WORLD_STATE_DIM))
        assert not torch.allclose(d.rsample(), d.rsample())


# ===================================================================
# 2. Deterministic mode and gradients
# ===================================================================

class TestDeterministicAndGradients:
    def test_mode_repeatable(self, ac):
        c = torch.randn(4, WORLD_STATE_DIM)
        torch.testing.assert_close(ac.actor(c).mode(), ac.actor(c).mode())

    def test_rsample_has_gradients(self, ac):
        c_t = torch.randn(4, WORLD_STATE_DIM, requires_grad=True)
        loss = ac.actor(c_t).rsample().sum()
        loss.backward()
        assert c_t.grad is not None and c_t.grad.abs().sum().item() > 0


# ===================================================================
# 3. log-probability
# ===================================================================

class TestLogProb:
    def test_finite_near_bounds(self, ac):
        c_t = torch.randn(16, WORLD_STATE_DIM)
        dist = ac.actor(c_t)
        near = torch.randn(16, 3)
        near[:, 0] = near[:, 0].clamp(-1, 1) * 0.999
        near[:, 1] = near[:, 1].sigmoid().clamp(0.001, 0.999)
        near[:, 2] = near[:, 2].sigmoid().clamp(0.001, 0.999)
        assert torch.isfinite(dist.log_prob(near)).all()

    def test_in_plausible_range(self, ac):
        dist = ac.actor(torch.randn(8, WORLD_STATE_DIM))
        assert dist.log_prob(dist.rsample()).mean().item() > -20.0

    def test_steer_gas_independent(self, ac):
        c_t = torch.randn(1, WORLD_STATE_DIM)
        dist = ac.actor(c_t)
        lp = dist.log_prob
        assert lp(torch.tensor([[0.0, 0.5, 0.5]])).item() != lp(
            torch.tensor([[0.5, 0.5, 0.5]])
        ).item()


# ===================================================================
# 4. λ-returns with termination/continuation contract
# ===================================================================

class TestLambdaReturns:
    def test_no_discount_single_step(self):
        """G_0 = r_0 (last step, continuation=False)."""
        tc = _tc(1)
        tc["continuation"][:, -1] = False
        ret = compute_lambda_returns(
            torch.tensor([[5.0]]), torch.tensor([[0.0]]),
            gamma=1.0, lambda_=0.0, **tc,
        )
        torch.testing.assert_close(ret, torch.tensor([[5.0]]))

    def test_single_step_with_bootstrap(self):
        """Last step continuation=True uses bootstrap_value."""
        tc = _tc(1)
        tc["continuation"][:, -1] = True
        tc["bootstrap_value"] = torch.tensor([10.0])
        ret = compute_lambda_returns(
            torch.tensor([[5.0]]), torch.tensor([[99.0]]),
            gamma=0.9, lambda_=0.5, **tc,
        )
        # G_0 = 5.0 + 0.9 * 10.0 = 14.0
        torch.testing.assert_close(ret, torch.tensor([[14.0]]), atol=1e-5, rtol=1e-5)

    def test_lambda_zero(self):
        """λ=0: G_t = r_t + γ * cont_t * V(s_{t+1})."""
        tc = _tc(3)
        ret = compute_lambda_returns(
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([[0.5, 0.4, 0.3]]),
            gamma=0.9, lambda_=0.0, **tc,
        )
        # G_2 = 3.0 + 0.9 * 0.0 = 3.0   (bootstrap_value=0)
        # G_1 = 2.0 + 0.9 * 0.3 = 2.27
        # G_0 = 1.0 + 0.9 * 0.4 = 1.36
        torch.testing.assert_close(
            ret, torch.tensor([[1.36, 2.27, 3.0]]), atol=1e-5, rtol=1e-5,
        )

    def test_terminated_mid_sequence(self):
        """terminated at t=1 → continuation=False, no bootstrap."""
        tc = _tc(3)
        tc["terminated"][:, 1] = True
        tc["continuation"][:, 1] = False
        ret = compute_lambda_returns(
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([[0.5, 0.4, 0.3]]),
            gamma=0.9, lambda_=0.0, **tc,
        )
        # G_2 = 3.0
        # G_1 = 2.0 + 0 = 2.0  (continuation=False)
        # G_0 = 1.0 + 0.9*0.4 = 1.36
        torch.testing.assert_close(
            ret, torch.tensor([[1.36, 2.0, 3.0]]), atol=1e-5, rtol=1e-5,
        )

    def test_truncated_continuation(self):
        """Truncated (terminated=False, continuation=True) → bootstrap used."""
        tc = _tc(3)
        tc["continuation"][:, -1] = True
        tc["bootstrap_value"] = torch.tensor([7.0])
        ret = compute_lambda_returns(
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([[0.5, 0.4, 0.3]]),
            gamma=0.9, lambda_=0.0, **tc,
        )
        # G_2 = 3.0 + 0.9*7.0 = 9.3
        # G_1 = 2.0 + 0.9*0.3 = 2.27
        # G_0 = 1.0 + 0.9*0.4 = 1.36
        torch.testing.assert_close(
            ret, torch.tensor([[1.36, 2.27, 9.3]]), atol=1e-5, rtol=1e-5,
        )

    def test_lambda_one_full_mc(self):
        """λ=1: G_t = r_t + γ * cont_t * G_{t+1}."""
        tc = _tc(3)
        ret = compute_lambda_returns(
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([[0.5, 0.4, 0.3]]),
            gamma=0.9, lambda_=1.0, **tc,
        )
        # G_2 = 3.0
        # G_1 = 2.0 + 0.9*3.0 = 4.7
        # G_0 = 1.0 + 0.9*4.7 = 5.23
        torch.testing.assert_close(
            ret, torch.tensor([[5.23, 4.7, 3.0]]), atol=1e-5, rtol=1e-5,
        )

    def test_two_step_hand_calculated(self):
        tc = _tc(2)
        tc["terminated"][:, 1] = True
        tc["continuation"][:, 1] = False
        ret = compute_lambda_returns(
            torch.tensor([[5.0, -1.0]]),
            torch.tensor([[0.0, 0.0]]),
            gamma=0.9, lambda_=0.5, **tc,
        )
        # G_1 = -1.0 (terminated)
        # G_0 = 5.0 + 0.9*(0.5*0.0 + 0.5*(-1.0)) = 5.0 - 0.45 = 4.55
        torch.testing.assert_close(
            ret, torch.tensor([[4.55, -1.0]]), atol=1e-5, rtol=1e-5,
        )


# ===================================================================
# 5. Termination / continuation semantics
# ===================================================================

class TestTerminationSemantics:
    def test_terminated_no_bootstrap(self):
        """terminated=True → continuation must be False, G=r."""
        rewards = torch.randn(2, 4)
        boot = torch.randn(2, 4)
        term = torch.zeros(2, 4, dtype=torch.bool)
        term[:, 2] = True
        cont = torch.ones(2, 4, dtype=torch.bool)
        cont[:, 2] = False
        bv = torch.zeros(2)
        ret = compute_lambda_returns(
            rewards, boot, term, cont, bv, gamma=0.9, lambda_=0.0,
        )
        assert torch.isfinite(ret).all()
        torch.testing.assert_close(ret[:, 2], rewards[:, 2])

    def test_truncated_final_bootstrap(self):
        """Truncated → continuation=True at last step, bootstrap_value used."""
        rewards = torch.tensor([[0.0, 1.0]])
        boot = torch.tensor([[5.0, 5.0]])
        term = torch.zeros(1, 2, dtype=torch.bool)
        cont = torch.ones(1, 2, dtype=torch.bool)
        bv = torch.tensor([10.0])
        ret = compute_lambda_returns(
            rewards, boot, term, cont, bv, gamma=0.9, lambda_=0.0,
        )
        # G_1 = 1.0 + 0.9*10.0 = 10.0
        # G_0 = 0.0 + 0.9*5.0 = 4.5
        torch.testing.assert_close(
            ret, torch.tensor([[4.5, 10.0]]), atol=1e-5, rtol=1e-5,
        )

    def test_explicit_no_continuation_end(self):
        """A deliberately non-bootstrap boundary has reward-only return.

        This is not the normal imagined-horizon case: an imagined rollout
        creates its final latent state, so Stage 5 sets continuation=True and
        supplies its target-critic bootstrap value.
        """
        rewards = torch.tensor([[2.0, 3.0]])
        boot = torch.tensor([[1.0, 1.0]])
        term = torch.zeros(1, 2, dtype=torch.bool)
        cont = torch.ones(1, 2, dtype=torch.bool)
        cont[:, -1] = False
        bv = torch.tensor([999.0])  # should be ignored
        ret = compute_lambda_returns(
            rewards, boot, term, cont, bv, gamma=0.9, lambda_=0.0,
        )
        # G_1 = 3.0 (cont=False, no bootstrap)
        # G_0 = 2.0 + 0.9*1.0 = 2.9
        torch.testing.assert_close(
            ret, torch.tensor([[2.9, 3.0]]), atol=1e-5, rtol=1e-5,
        )

    def test_imagined_horizon_with_final_state_bootstraps(self):
        """A normal imagined horizon bootstraps from its final latent state."""
        rewards = torch.tensor([[2.0, 3.0]])
        boot = torch.tensor([[1.0, 1.0]])
        term = torch.zeros(1, 2, dtype=torch.bool)
        cont = torch.ones(1, 2, dtype=torch.bool)
        final_value = torch.tensor([4.0])
        ret = compute_lambda_returns(
            rewards, boot, term, cont, final_value, gamma=0.9, lambda_=0.0,
        )
        # G_1 = 3 + .9 * V_target(s_2) = 6.6; G_0 = 2 + .9 * 1 = 2.9.
        torch.testing.assert_close(
            ret, torch.tensor([[2.9, 6.6]]), atol=1e-5, rtol=1e-5,
        )


# ===================================================================
# 6. Critic overfits synthetic returns
# ===================================================================

class TestCriticOverfit:
    @pytest.mark.models
    def test_critic_overfits(self, ac):
        B, T, D = 8, 4, WORLD_STATE_DIM
        beliefs = torch.randn(B, T, D)
        targets = beliefs.sum(dim=-1) * 0.01
        c_t = ac.encode(beliefs.reshape(B * T, D)).reshape(B, T, -1)
        opt = torch.optim.Adam(ac.critic.parameters(), lr=1e-2)
        for step in range(500):
            v = ac.critic(c_t.reshape(B * T, -1)).reshape(B, T)
            loss = ((v - targets) ** 2).mean()
            if step == 0:
                init_loss = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        final_v = ac.critic(c_t.reshape(B * T, -1)).reshape(B, T)
        final_mse = ((final_v - targets) ** 2).mean().item()
        assert final_mse < init_loss / 2.0

    @pytest.mark.models
    def test_optimizer_step_reduces_critic_loss(self, ac):
        B, T, D = 4, 4, WORLD_STATE_DIM
        beliefs = torch.randn(B, T, D)
        actions = torch.zeros(B, T, ACTION_DIM)
        actions[..., 0] = torch.rand(B, T) * 2 - 1
        actions[..., 1:] = torch.rand(B, T, 2)
        rewards = torch.randn(B, T)
        term = torch.zeros(B, T, dtype=torch.bool)
        cont = torch.ones(B, T, dtype=torch.bool)
        bv = torch.zeros(B)
        losses = []
        for _ in range(50):
            m = ac.optimizer_step(
                beliefs, actions, rewards, term, cont, bv,
                gamma=0.9, lambda_=0.0,
            )
            losses.append(m["critic_loss"])
        assert losses[-1] < losses[0] * 0.8


# ===================================================================
# 7. Advantage direction
# ===================================================================

class TestAdvantageDirection:
    @pytest.mark.models
    def _train_one(self, ac, beliefs, actions, rewards, entropy_coef=0.0):
        B, T = beliefs.shape[0], beliefs.shape[1]
        term = torch.zeros(B, T, dtype=torch.bool)
        cont = torch.ones(B, T, dtype=torch.bool)
        bv = torch.zeros(B)
        for _ in range(30):
            ac.optimizer_step(
                beliefs, actions, rewards, term, cont, bv,
                gamma=0.9, lambda_=0.0, entropy_coef=entropy_coef,
            )

    @pytest.mark.models
    def test_positive_advantage_increases_log_prob(self, ac):
        B, D = 2, WORLD_STATE_DIM
        beliefs = torch.randn(B, 1, D)
        actions = ac.actor(ac.encode(beliefs[:, 0])).mode().unsqueeze(1)
        with torch.no_grad():
            lp_before = ac.actor(ac.encode(beliefs[:, 0])).log_prob(
                actions[:, 0]
            )
        self._train_one(ac, beliefs, actions, torch.ones(B, 1))
        with torch.no_grad():
            lp_after = ac.actor(ac.encode(beliefs[:, 0])).log_prob(
                actions[:, 0]
            )
        assert (lp_after > lp_before).all()

    @pytest.mark.models
    def test_negative_advantage_decreases_log_prob(self, ac):
        B, D = 2, WORLD_STATE_DIM
        beliefs = torch.randn(B, 1, D)
        actions = ac.actor(ac.encode(beliefs[:, 0])).mode().unsqueeze(1)
        with torch.no_grad():
            lp_before = ac.actor(ac.encode(beliefs[:, 0])).log_prob(
                actions[:, 0]
            )
        self._train_one(ac, beliefs, actions, -torch.ones(B, 1))
        with torch.no_grad():
            lp_after = ac.actor(ac.encode(beliefs[:, 0])).log_prob(
                actions[:, 0]
            )
        assert (lp_after < lp_before).all()


# ===================================================================
# 8. Entropy coefficient direction
# ===================================================================

class TestEntropyDirection:
    @pytest.mark.models
    def test_positive_entropy_increases_entropy(self, ac):
        B, T, D = 4, 2, WORLD_STATE_DIM
        beliefs = torch.randn(B, T, D)
        acts = torch.randn(B, T, ACTION_DIM)
        acts[..., 0] = acts[..., 0].clamp(-1, 1)
        acts[..., 1:] = acts[..., 1:].sigmoid()
        rewards = torch.zeros(B, T)
        term = torch.zeros(B, T, dtype=torch.bool)
        cont = torch.ones(B, T, dtype=torch.bool)
        bv = torch.zeros(B)

        # Phase 1: entropy_coef=0
        for _ in range(20):
            ac.optimizer_step(
                beliefs, acts, rewards, term, cont, bv,
                gamma=0.9, lambda_=0.0, entropy_coef=0.0,
            )
        with torch.no_grad():
            ent_low = ac.actor(ac.encode(beliefs[:, 0])).entropy().mean()

        # Phase 2: higher entropy
        for _ in range(40):
            ac.optimizer_step(
                beliefs, acts, rewards, term, cont, bv,
                gamma=0.9, lambda_=0.0, entropy_coef=0.05,
            )
        with torch.no_grad():
            ent_high = ac.actor(ac.encode(beliefs[:, 0])).entropy().mean()

        assert ent_high > ent_low


# ===================================================================
# 9. Target Critic Polyak update
# ===================================================================

class TestTargetCritic:
    def test_hard_copy_at_init(self, ac):
        for t, o in zip(ac.target_critic.parameters(), ac.critic.parameters()):
            torch.testing.assert_close(t, o)

    def test_polyak_update_blends(self, ac):
        t_before = [p.data.clone() for p in ac.target_critic.parameters()]
        for p in ac.critic.parameters():
            p.data.add_(torch.randn_like(p) * 0.1)
        tau = 0.1
        ac._sync_target(tau=tau)
        for i, (t_p, o_p) in enumerate(
            zip(ac.target_critic.parameters(), ac.critic.parameters())
        ):
            expected = tau * o_p + (1 - tau) * t_before[i]
            torch.testing.assert_close(t_p, expected)


# ===================================================================
# 10. Freeze contract
# ===================================================================

class TestFreezeContract:
    @pytest.mark.models
    def test_world_model_params_unchanged(self, ac):
        B, T, D = 4, 4, WORLD_STATE_DIM
        beliefs = torch.randn(B, T, D)
        acts = torch.zeros(B, T, ACTION_DIM)
        acts[..., 0] = torch.rand(B, T) * 2 - 1
        acts[..., 1:] = torch.rand(B, T, 2)
        r = torch.randn(B, T)
        term = torch.zeros(B, T, dtype=torch.bool)
        cont = torch.ones(B, T, dtype=torch.bool)
        bv = torch.zeros(B)
        snap = {k: v.data.clone() for k, v in ac.world_model.state_dict().items()}
        for _ in range(10):
            ac.optimizer_step(beliefs, acts, r, term, cont, bv)
        for k, v in ac.world_model.state_dict().items():
            torch.testing.assert_close(v, snap[k])

    def test_world_model_frozen(self, ac):
        assert all(not p.requires_grad for p in ac.world_model.parameters())

    def test_optimizers_only_ac_params(self, ac):
        ac_params = set(ac.actor.parameters()) | set(ac.critic.parameters())
        for g in ac._actor_optim.param_groups:
            assert all(p in ac_params for p in g["params"])
        for g in ac._critic_optim.param_groups:
            assert all(p in ac_params for p in g["params"])


# ===================================================================
# 11. Validation — terminated/continuation conflict
# ===================================================================

class TestValidation:
    def test_terminated_continuation_conflict_raises(self):
        term = torch.tensor([[False, True, False]])
        cont = torch.tensor([[False, True, True]])
        with pytest.raises(ValueError, match="conflict"):
            _validate_tc(term, cont)

    def test_non_bool_terminated_raises(self):
        with pytest.raises(TypeError):
            _validate_tc(
                torch.tensor([[0, 1, 0]]),
                torch.ones(1, 3, dtype=torch.bool),
            )

    def test_non_bool_continuation_raises(self):
        with pytest.raises(TypeError):
            _validate_tc(
                torch.zeros(1, 3, dtype=torch.bool),
                torch.tensor([[1, 1, 1]]),
            )

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            _validate_tc(
                torch.zeros(1, 4, dtype=torch.bool),
                torch.ones(2, 4, dtype=torch.bool),
            )

    def test_bootstrap_value_shape_raises(self):
        rewards = torch.zeros(2, 3)
        boot = torch.zeros(2, 3)
        term = torch.zeros(2, 3, dtype=torch.bool)
        cont = torch.ones(2, 3, dtype=torch.bool)
        with pytest.raises(ValueError, match="bootstrap_value"):
            compute_lambda_returns(rewards, boot, term, cont, torch.zeros(1))

    def test_optimizer_step_validates(self, ac):
        B, T, D = 2, 4, WORLD_STATE_DIM
        z = torch.randn(B, T, D)
        a = torch.randn(B, T, ACTION_DIM)
        r = torch.randn(B, T)
        term = torch.zeros(B, T, dtype=torch.bool)
        cont = torch.ones(B, T, dtype=torch.bool)
        term[:, 1] = True
        cont[:, 1] = True  # conflict
        bv = torch.zeros(B)
        with pytest.raises(ValueError, match="conflict"):
            ac.optimizer_step(z, a, r, term, cont, bv)

    def test_lambda_returns_validates(self):
        r = torch.randn(1, 3)
        b = torch.randn(1, 3)
        term = torch.tensor([[False, True, False]])
        cont = torch.tensor([[False, True, True]])  # conflict at 1
        bv = torch.zeros(1)
        with pytest.raises(ValueError, match="conflict"):
            compute_lambda_returns(r, b, term, cont, bv)

    def test_td_advantage_validates(self):
        v = torch.randn(1, 3)
        r = torch.randn(1, 3)
        b = torch.randn(1, 3)
        term = torch.tensor([[False, True, False]])
        cont = torch.tensor([[False, True, True]])
        bv = torch.zeros(1)
        with pytest.raises(ValueError, match="conflict"):
            compute_td_advantage(v, r, b, term, cont, bv)


# ===================================================================
# 12. BoundedGaussian.entropy_sample
# ===================================================================

class TestEntropySample:
    def test_entropy_sample_finite(self, ac):
        dist = ac.actor(torch.randn(8, WORLD_STATE_DIM))
        es = dist.entropy_sample()
        assert torch.isfinite(es).all()
        assert es.shape == (8,)

    def test_entropy_sample_has_gradients(self, ac):
        ac.actor.train()
        c_t = torch.randn(4, WORLD_STATE_DIM, requires_grad=True)
        dist = ac.actor(c_t)
        loss = dist.entropy_sample().mean()
        loss.backward()
        assert c_t.grad is not None
        assert c_t.grad.abs().sum().item() > 0

    def test_entropy_sample_differs_from_raw(self, ac):
        """Sampled entropy should differ from raw-Gaussian approximation."""
        c_t = torch.randn(8, WORLD_STATE_DIM)
        dist = ac.actor(c_t)
        raw = dist.entropy()
        sampled = dist.entropy_sample()
        assert not torch.allclose(raw, sampled, atol=1e-3)

    def test_entropy_sample_repeatable_stochastic(self, ac):
        """Different calls to entropy_sample give different values."""
        c_t = torch.randn(4, WORLD_STATE_DIM)
        dist = ac.actor(c_t)
        e1 = dist.entropy_sample()
        e2 = dist.entropy_sample()
        assert not torch.allclose(e1, e2)


# ===================================================================
# 13. Config round-trip
# ===================================================================

class TestConfig:
    def test_defaults(self):
        c = ActorCriticConfig()
        assert c.hidden_dim == 64 and c.entropy_coef == 1e-3

    def test_round_trip(self):
        c = ActorCriticConfig(hidden_dim=128, entropy_coef=0.01)
        assert ActorCriticConfig.from_dict(c.to_dict()).hidden_dim == 128

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            ActorCriticConfig().hidden_dim = 128
