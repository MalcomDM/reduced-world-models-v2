"""Tests for the corrected KL reduction (Stage 2.5B.1).

Mathematical contract:

    For posterior tensors mu and logvar with shape (B, T, P, D):

    kl_per_element = 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
    kl = kl_per_element.mean()

The nonlinear KL expression must be applied **before** any averaging over
time, patches, dimensions, or batch.  Averaging mu or logvar first (as the
old code did) gives a mathematically different result.
"""

import pytest
import torch

from rwm.trainers.deterministic.world_model_trainer import kl_normal


# ===================================================================
# Zero / unit posterior tests
# ===================================================================

class TestKnownValues:
    def test_zero_mu_logvar_gives_zero_kl(self):
        """KL(N(0,1) || N(0,1)) = 0 for any shape."""
        mu = torch.zeros(2, 4, 225, 16)
        logvar = torch.zeros(2, 4, 225, 16)
        kl = kl_normal(mu, logvar)
        assert kl.item() == 0.0

    def test_known_posterior_gives_expected_kl(self):
        """For mu=1, logvar=0 (sigma=1):
        KL = 0.5 * (1^2 + exp(0) - 1 - 0) = 0.5 * (1 + 1 - 1) = 0.5
        """
        mu = torch.ones(2, 3, 225, 16)
        logvar = torch.zeros(2, 3, 225, 16)
        kl = kl_normal(mu, logvar)
        assert kl.item() == 0.5

    def test_known_logvar_gives_expected_kl(self):
        """For mu=2, logvar=1 (sigma=exp(0.5)):
        KL = 0.5 * (2^2 + exp(1) - 1 - 1)
           = 0.5 * (4 + 2.7183 - 2)
           = 0.5 * 4.7183
           = 2.35915
        """
        mu = torch.full((1, 1, 1, 1), 2.0)
        logvar = torch.full((1, 1, 1, 1), 1.0)
        kl = kl_normal(mu, logvar)
        expected = 0.5 * (4.0 + torch.exp(torch.tensor(1.0)) - 1.0 - 1.0)
        assert abs(kl.item() - expected.item()) < 1e-5


# ===================================================================
# Pre-averaging regression test
# ===================================================================

class TestPreAveragingRegression:
    def test_averaging_mu_before_kl_is_wrong(self):
        """Prove that computing KL on time-averaged posterior gives a
        different (incorrect) result compared to per-element KL.

        Two timesteps with different posteriors:
          t=0: mu=2, logvar=0
          t=1: mu=0, logvar=0

        Per-element KL:
          t=0: 0.5 * (4 + 1 - 1 - 0) = 2.0
          t=1: 0.5 * (0 + 1 - 1 - 0) = 0.0
          mean = 1.0

        Averaged posterior: mu=1, logvar=0
          KL = 0.5 * (1 + 1 - 1 - 0) = 0.5
        """
        B, T, P, D = 1, 2, 1, 1
        mu = torch.zeros(B, T, P, D)
        mu[:, 0] = 2.0
        mu[:, 1] = 0.0
        logvar = torch.zeros(B, T, P, D)

        # Correct per-element KL
        per_element = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        correct_kl = per_element.mean()

        # Wrong: average posterior first
        mu_avg = mu.mean(dim=1, keepdim=True)  # (B, 1, P, D)
        logvar_avg = logvar.mean(dim=1, keepdim=True)
        wrong_kl_value = 0.5 * (mu_avg.pow(2) + logvar_avg.exp() - 1.0 - logvar_avg).mean()

        assert abs(correct_kl.item() - 1.0) < 1e-6, f"Expected correct KL=1.0, got {correct_kl.item()}"
        assert abs(wrong_kl_value.item() - 0.5) < 1e-6, f"Expected wrong KL=0.5, got {wrong_kl_value.item()}"
        assert correct_kl.item() > wrong_kl_value.item(), (
            "Pre-averaging KL should give a smaller (incorrect) result"
        )

        # The kl_normal function must match the correct per-element result
        computed_kl = kl_normal(mu, logvar)
        assert abs(computed_kl.item() - correct_kl.item()) < 1e-6, (
            f"kl_normal returned {computed_kl.item()}, expected {correct_kl.item()}"
        )

    def test_regression_vs_old_behavior(self):
        """On time-constant posteriors, old and new KL match.

        When all timesteps have the same posterior, pre-averaging and
        per-element KL produce the same result (because the KL is
        linear in the sufficient statistics).
        """
        B, T, P, D = 3, 5, 225, 16
        mu = torch.randn(B, 1, P, D).expand(-1, T, -1, -1).clone()
        logvar = torch.randn(B, 1, P, D).expand(-1, T, -1, -1).clone()

        # Compute using the corrected (per-element) approach
        per_element = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        correct_kl = per_element.mean()

        computed_kl = kl_normal(mu, logvar)
        assert abs(computed_kl.item() - correct_kl.item()) < 1e-6

        # This is exactly the legacy model path when no frame differs.
        old_kl = kl_normal(mu.mean(dim=1), logvar.mean(dim=1))
        assert torch.allclose(computed_kl, old_kl)


# ===================================================================
# Shape-agnostic test
# ===================================================================

class TestShapeAgnostic:
    def test_single_frame_shape(self):
        """Single frame: (B, P, D) — no time dimension."""
        mu = torch.randn(2, 225, 16)
        logvar = torch.randn(2, 225, 16)
        kl = kl_normal(mu, logvar)
        assert kl.ndim == 0  # scalar
        assert kl.item() >= 0

    def test_sequence_shape(self):
        """Full sequence: (B, T, P, D)."""
        mu = torch.randn(2, 5, 225, 16)
        logvar = torch.randn(2, 5, 225, 16)
        kl = kl_normal(mu, logvar)
        assert kl.ndim == 0
        assert kl.item() >= 0

    def test_extra_dimensions(self):
        """Higher-dimensional input should also work."""
        mu = torch.randn(2, 3, 4, 5, 6)  # (B, T, P, D, extra)
        logvar = torch.randn(2, 3, 4, 5, 6)
        kl = kl_normal(mu, logvar)
        assert kl.ndim == 0
        assert kl.item() >= 0


# ===================================================================
# Gradient test
# ===================================================================

class TestGradient:
    def test_kl_gives_gradients_with_nonzero_beta(self):
        """KL must produce finite nonzero gradients for mu/logvar."""
        mu = torch.randn(2, 4, 225, 16, requires_grad=True)
        logvar = torch.randn(2, 4, 225, 16, requires_grad=True)
        beta = torch.tensor(1.0)

        kl = kl_normal(mu, logvar)
        loss = beta * kl
        loss.backward()

        assert mu.grad is not None
        assert logvar.grad is not None
        assert mu.grad.abs().sum().item() > 0
        assert logvar.grad.abs().sum().item() > 0
        assert not torch.isnan(mu.grad).any()
        assert not torch.isnan(logvar.grad).any()


# ===================================================================
# Sequence dependence test
# ===================================================================

class TestSequenceDependence:
    def test_changing_one_timestep_affects_kl(self):
        """Changing mu/logvar at one timestep must change the KL result
        (proves temporal dimension is included)."""
        B, T, P, D = 1, 5, 10, 4
        mu = torch.randn(B, T, P, D)
        logvar = torch.randn(B, T, P, D)

        kl1 = kl_normal(mu, logvar)

        # Change one timestep
        mu2 = mu.clone()
        mu2[:, 2] = 999.0
        kl2 = kl_normal(mu2, logvar)

        assert abs(kl1.item() - kl2.item()) > 0.01, (
            "Changing one timestep should change KL"
        )


def test_forward_sequence_preserves_posterior_time_axis():
    """The model output, not only the KL helper, must retain every frame."""
    from rwm.models.rwm.model import ReducedWorldModel

    model = ReducedWorldModel().eval()
    obs = torch.zeros(1, 2, 3, 64, 64)
    prev_actions = torch.zeros(1, 2, 3)
    current_actions = torch.zeros(1, 2, 3)

    with torch.no_grad():
        out = model.forward_sequence(obs, prev_actions, current_actions, force_keep_input=True)

    assert out.tok_mu is not None
    assert out.tok_logvar is not None
    assert out.tok_mu.shape == (1, 2, 225, 16)
    assert out.tok_logvar.shape == (1, 2, 225, 16)
