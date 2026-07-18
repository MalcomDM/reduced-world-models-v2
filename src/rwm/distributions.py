"""Reparameterizable bounded action distributions for continuous control.

All distributions maintain gradient flow through ``rsample()`` and provide
correct change-of-variables handling for log-probability computation.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor


class BoundedGaussian:
    """Reparameterized diagonal Gaussian with per-dimension squashing.

    Dim 0 (steering): ``tanh`` → ``[-1, 1]``.
    Dims 1, 2 (gas, brake): ``sigmoid`` → ``[0, 1]``.

    Provides ``rsample()`` (reparameterized), ``mode()`` (deterministic),
    ``log_prob()`` (with change-of-variables correction), and ``entropy()``
    (raw Gaussian approximation).

    Parameters
    ----------
    mean:
        ``(B, A)`` — location parameters.
    logstd:
        ``(B, A)`` — log-standard-deviations (clamped to ``[-10, 2]``
        internally for numerical stability).
    """

    def __init__(self, mean: Tensor, logstd: Tensor) -> None:
        self.mean = mean
        self.logstd = logstd.clamp(-10.0, 2.0)
        self.std = self.logstd.exp()

    # ------------------------------------------------------------------
    # Squashing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _squash(z: Tensor) -> Tensor:
        """Raw sample ``z (B, 3)`` → bounded action ``a (B, 3)``."""
        steer = torch.tanh(z[..., 0:1])
        gas = torch.sigmoid(z[..., 1:2])
        brake = torch.sigmoid(z[..., 2:3])
        return torch.cat([steer, gas, brake], dim=-1)

    @staticmethod
    def _inverse(a: Tensor) -> Tensor:
        """Bounded action ``a (B, 3)`` → raw latent ``z (B, 3)``."""
        steer_z = torch.arctanh(a[..., 0:1].clamp(-0.9999999, 0.9999999))
        gas_z = torch.logit(a[..., 1:2].clamp(1e-7, 1.0 - 1e-7))
        brake_z = torch.logit(a[..., 2:3].clamp(1e-7, 1.0 - 1e-7))
        return torch.cat([steer_z, gas_z, brake_z], dim=-1)

    @staticmethod
    def _jacobian_logdet(z: Tensor) -> Tensor:
        """Log-abs-determinant of the squash Jacobian ``(B,)``.

        Summed over action dims::

            dim 0 (tanh):  log|1 - tanh(z)²|
            dims 1, 2 (sigmoid): log|sigmoid(z) * (1 - sigmoid(z))|
        """
        steer_z = z[..., 0]
        steer_out = torch.tanh(steer_z)
        log_det = torch.log(1.0 - steer_out ** 2 + 1e-6)

        for i in (1, 2):
            out = torch.sigmoid(z[..., i])
            log_det = log_det + torch.log(out * (1.0 - out) + 1e-6)

        return log_det

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rsample(self) -> Tensor:
        """Reparameterized sample ``(B, A)``.  Gradients flow through."""
        eps = torch.randn_like(self.mean)
        z = self.mean + eps * self.std
        return self._squash(z)

    def mode(self) -> Tensor:
        """Deterministic mode ``(B, A)``."""
        return self._squash(self.mean)

    def log_prob(self, action: Tensor) -> Tensor:
        """Log-probability ``(B,)`` with change-of-variables correction."""
        z = self._inverse(action)
        err = (z - self.mean) / self.std
        log_prob_z = -0.5 * (
            math.log(2.0 * math.pi) + 2.0 * self.logstd + err ** 2
        )
        log_prob_z = log_prob_z.sum(-1)
        return log_prob_z - self._jacobian_logdet(z)

    def entropy(self) -> Tensor:
        """Entropy estimate ``(B,)`` — raw-Gaussian approximation (ignores
        squashing).  This is a lower bound on the true entropy and is cheap
        to compute, but does not reflect the action bounds.

        For a Monte-Carlo estimate that accounts for the full transform, use
        ``entropy_sample()``.
        """
        return 0.5 * (1.0 + math.log(2.0 * math.pi)) + self.logstd.sum(-1)

    def entropy_sample(self) -> Tensor:
        """Single-sample Monte-Carlo entropy ``(B,)`` through the full
        squashed distribution.

        ``H(p) ≈ -log p(a)`` where ``a ~ p`` via ``rsample()``.  Gradients
        flow through both the sample path and the log-probability, enabling
        entropy-penalty gradients that affect the raw scale parameters.
        """
        return -self.log_prob(self.rsample())
