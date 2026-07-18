"""Actor-Critic heads for frozen-world-model control (Stage 4.0).

Architecture (calibration baseline)::

    c_t = ControllerTrunk.encode(z_t)   — 80-dim shared representation
    Actor(c_t) → action distribution     — 80 → 64 → (mean, logstd)
    Critic(c_t) → V(c_t)                 — 80 → 64 → scalar

The full ``ReducedWorldModel``, including ``ControllerTrunk`` and reward head,
is frozen during calibration.  Only Actor/Critic parameters are optimised.

Termination / continuation contract
------------------------------------

``terminated`` and ``continuation`` are two boolean flags per step:

- ``terminated[t] = True``
    The environment reached a true terminal state at step ``t``.
    **No bootstrap** from the next state. ``continuation[t]`` must be False.
    The return target is simply ``r_t``.

- ``terminated[t] = False, continuation[t] = True``
    A valid next state exists.  Standard λ-return bootstrap applies.

- ``terminated[t] = False, continuation[t] = False``
    End of an artificial imagined horizon; no next-state value is available.
    The return target is simply ``r_t`` (no bootstrap).

- ``terminated[t] = True, continuation[t] = True``
    **Invalid** — raises ``ValueError``.

The final-step bootstrap value ``V(s_T)`` is supplied separately as
``bootstrap_value``, used only when ``continuation[:, -1]`` is True.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from rwm.config.config import ACTION_DIM, WORLD_STATE_DIM
from rwm.config.experiment_config import ActorCriticConfig
from rwm.distributions import BoundedGaussian


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_tc(
    terminated: Tensor,
    continuation: Tensor,
) -> None:
    """Validate the termination / continuation contract.

    Raises ``ValueError`` if any position has both flags True.
    """
    if terminated.dtype != torch.bool:
        raise TypeError(
            f"terminated must be bool, got {terminated.dtype}"
        )
    if continuation.dtype != torch.bool:
        raise TypeError(
            f"continuation must be bool, got {continuation.dtype}"
        )
    conflict = terminated & continuation
    if conflict.any():
        idx = conflict.nonzero(as_tuple=False)[0].tolist()
        raise ValueError(
            f"conflict at {idx}: terminated=True and continuation=True "
            "are not allowed simultaneously"
        )
    if terminated.shape != continuation.shape:
        raise ValueError(
            f"terminated shape {terminated.shape} does not match "
            f"continuation shape {continuation.shape}"
        )


def _validate_return_inputs(
    rewards: Tensor,
    bootstrap_values: Tensor,
    terminated: Tensor,
    continuation: Tensor,
    bootstrap_value: Tensor,
) -> None:
    """Validate return tensors before arithmetic can broadcast silently."""
    _validate_tc(terminated, continuation)
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape (B, T), got {rewards.shape}")
    if bootstrap_values.shape != rewards.shape:
        raise ValueError(
            "bootstrap_values shape "
            f"{bootstrap_values.shape} does not match rewards {rewards.shape}"
        )
    if terminated.shape != rewards.shape:
        raise ValueError(
            f"terminated shape {terminated.shape} does not match rewards {rewards.shape}"
        )
    expected_bootstrap_shape = (rewards.shape[0],)
    if bootstrap_value.shape != expected_bootstrap_shape:
        raise ValueError(
            "bootstrap_value must have shape "
            f"{expected_bootstrap_shape}, got {bootstrap_value.shape}"
        )
    tensors = (bootstrap_values, terminated, continuation, bootstrap_value)
    if any(t.device != rewards.device for t in tensors):
        raise ValueError("return tensors must all be on the rewards device")


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Policy head: shared representation → action distribution.

    Parameters
    ----------
    input_dim:
        Dimension of ``c_t`` (default ``WORLD_STATE_DIM = 80``).
    hidden_dim:
        Hidden layer width.
    action_dim:
        Number of action dimensions.
    """

    def __init__(
        self,
        input_dim: int = WORLD_STATE_DIM,
        hidden_dim: int = 64,
        action_dim: int = ACTION_DIM,
        min_logstd: float = -10.0,
        max_logstd: float = 2.0,
    ) -> None:
        super().__init__()
        self._hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self._mean_head = nn.Linear(hidden_dim, action_dim)
        self._logstd_head = nn.Linear(hidden_dim, action_dim)
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd

    def forward(self, c_t: Tensor) -> BoundedGaussian:
        """Build action distribution from shared representation ``c_t (B, D)``."""
        h = self._hidden(c_t)
        mean = self._mean_head(h)
        logstd = self._logstd_head(h).clamp(self._min_logstd, self._max_logstd)
        return BoundedGaussian(mean, logstd)


class Critic(nn.Module):
    """Value head: shared representation → scalar V(c_t).

    Parameters
    ----------
    input_dim:
        Dimension of ``c_t`` (default ``WORLD_STATE_DIM = 80``).
    hidden_dim:
        Hidden layer width.
    """

    def __init__(
        self,
        input_dim: int = WORLD_STATE_DIM,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, c_t: Tensor) -> Tensor:
        """``(B, 1)`` scalar value estimate."""
        return self._net(c_t)


# ---------------------------------------------------------------------------
# Combined module
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Combined Actor-Critic module with frozen world model.

    Owns the Actor, online Critic, and target Critic.  Provides a single
    ``optimizer_step()`` that enforces the freeze contract: only Actor/Critic
    parameters are updated.

    Parameters
    ----------
    model:
        A ``ReducedWorldModel`` (or any module providing a ``controller``
        with an ``encode()`` method).  Its parameters are frozen externally
        or by this module.
    cfg:
        ``ActorCriticConfig`` with hyper-parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Optional[ActorCriticConfig] = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = ActorCriticConfig()
        self.cfg = cfg

        # World model — frozen.
        self.world_model = model
        self._freeze_world_model()

        # Heads.
        self.actor = Actor(hidden_dim=cfg.hidden_dim)
        self.critic = Critic(hidden_dim=cfg.hidden_dim)
        self.target_critic = Critic(hidden_dim=cfg.hidden_dim)
        self._sync_target( tau=1.0)  # hard copy

        # Optimisers — only actor/critic params.
        self._actor_optim: Optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr,
        )
        self._critic_optim: Optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr,
        )

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def _freeze_world_model(self) -> None:
        for p in self.world_model.parameters():
            p.requires_grad_(False)

    def freeze_world_model(self) -> None:
        """Public re-freeze (e.g. after accidental unfreeze)."""
        self._freeze_world_model()

    def freeze_actor_critic(self) -> None:
        """Freeze Actor and Critic (for eval)."""
        for p in self.actor.parameters():
            p.requires_grad_(False)
        for p in self.critic.parameters():
            p.requires_grad_(False)

    def unfreeze_actor_critic(self) -> None:
        """Unfreeze Actor and Critic for training."""
        for p in self.actor.parameters():
            p.requires_grad_(True)
        for p in self.critic.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------
    # Target Critic
    # ------------------------------------------------------------------

    def _sync_target(self, tau: Optional[float] = None) -> None:
        """Polyak update: ``target = τ * online + (1 - τ) * target``.

        When ``tau = 1.0``, performs a hard copy.
        """
        if tau is None:
            tau = self.cfg.target_update_rate
        with torch.no_grad():
            for t_param, o_param in zip(
                self.target_critic.parameters(),
                self.critic.parameters(),
            ):
                t_param.data.mul_(1.0 - tau).add_(o_param.data, alpha=tau)

    def update_target(self) -> None:
        """Call after each critic optimizer step."""
        self._sync_target()

    # ------------------------------------------------------------------
    # Representation extraction
    # ------------------------------------------------------------------

    def encode(self, z_t: Tensor) -> Tensor:
        """Shared representation ``c_t = ControllerTrunk.encode(z_t)``.

        The result is detached (world model is frozen and should not receive
        gradients from the Actor-Critic loss).
        """
        return self.world_model.controller.encode(z_t).detach()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z_t: Tensor) -> Tuple[BoundedGaussian, Tensor]:
        """Build action distribution and Critic value from belief ``z_t``.

        Equivalent to::

            c_t = self.encode(z_t)
            dist = self.actor(c_t)
            value = self.critic(c_t)
            return dist, value

        Returns
        -------
        dist:
            Action distribution for the current state.
        value:
            ``(B, 1)`` — online Critic value estimate.
        """
        c_t = self.encode(z_t)
        dist = self.actor(c_t)
        value = self.critic(c_t)
        return dist, value

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def compute_critic_loss(
        self,
        values: Tensor,               # (B, T)  online values
        target_values: Tensor,        # (B, T)  target values V(s_0…s_{T-1})
        rewards: Tensor,              # (B, T)
        terminated: Tensor,           # (B, T)  bool
        continuation: Tensor,         # (B, T)  bool
        bootstrap_value: Tensor,      # (B,)    V(s_T) for final-step bootstrap
        gamma: Optional[float] = None,
        lambda_: Optional[float] = None,
    ) -> Tensor:
        """Scalar MSE value loss against λ-returns.

        Parameters
        ----------
        values:
            Online Critic predictions ``V(s_t)``.
        target_values:
            Target Critic predictions ``V_target(s_0…s_{T-1})``; used as
            ``V(s_{t+1})`` during the bootstrap loop.
        rewards, terminated, continuation:
            Per-step signals (see module docstring for the contract).
        bootstrap_value:
            ``V_target(s_T)`` — used only when ``continuation[:, -1]`` is True.
        gamma:
            Discount factor (defaults to ``cfg.gamma``).
        lambda_:
            GAE-λ parameter (defaults to ``cfg.lambda_``).

        Returns
        -------
        ``()`` scalar loss.
        """
        if gamma is None:
            gamma = self.cfg.gamma
        if lambda_ is None:
            lambda_ = self.cfg.lambda_

        returns = compute_lambda_returns(
            rewards, target_values, terminated, continuation,
            bootstrap_value, gamma=gamma, lambda_=lambda_,
        )
        return nn.functional.mse_loss(values, returns.detach())

    def compute_actor_loss(
        self,
        dist: BoundedGaussian,
        actions: Tensor,       # (B*T, A) or (B, T, A)  actions taken
        advantage: Tensor,     # (B*T,) or (B, T)  advantage estimates
        entropy_coef: Optional[float] = None,
    ) -> Tensor:
        """Scalar actor loss: neg-log-prob weighted by advantage minus entropy.

        ``L = -mean(stopgrad(adv) * log_prob(action)) - entropy_coef * mean(entropy_sample)``

        Uses the sampled (Monte-Carlo) entropy estimate so gradients flow
        through both the log-prob and the reparameterized sample path.
        """
        if entropy_coef is None:
            entropy_coef = self.cfg.entropy_coef

        log_prob = dist.log_prob(actions)
        actor_loss = -(advantage.detach() * log_prob).mean()
        entropy = dist.entropy_sample().mean()
        return actor_loss - entropy_coef * entropy

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def optimizer_step(
        self,
        z_t: Tensor,                # (B, T, D)  beliefs
        actions: Tensor,            # (B, T, A)  actions taken
        rewards: Tensor,            # (B, T)  rewards
        terminated: Tensor,         # (B, T)  bool
        continuation: Tensor,       # (B, T)  bool
        bootstrap_value: Tensor,    # (B,)  V_target(s_T)
        gamma: Optional[float] = None,
        lambda_: Optional[float] = None,
        entropy_coef: Optional[float] = None,
    ) -> Dict[str, float]:
        """Single Actor-Critic optimisation step.

        Forward pass, compute losses, backprop, step optimisers, update
        target Critic.  World-model parameters are guaranteed unchanged.

        Parameters
        ----------
        z_t:
            Beliefs ``(B, T, D)`` from the world model.
        actions:
            ``(B, T, A)``.
        rewards:
            ``(B, T)``.
        terminated, continuation:
            ``(B, T)`` bool.  See module docstring for the contract.
        bootstrap_value:
            ``(B,)`` — target Critic value for the state after the last action.
            Used only when ``continuation[:, -1]`` is True.
        gamma, lambda_, entropy_coef:
            Overrides for config defaults.

        Returns
        -------
        ``{"actor_loss": ..., "critic_loss": ..., "entropy": ..., "value_mean": ...}``
        """
        if gamma is None:
            gamma = self.cfg.gamma
        if lambda_ is None:
            lambda_ = self.cfg.lambda_
        if entropy_coef is None:
            entropy_coef = self.cfg.entropy_coef

        _validate_tc(terminated, continuation)

        B, T, D = z_t.shape
        z_flat = z_t.reshape(B * T, D)

        # Detached representation.
        c_t = self.encode(z_flat).reshape(B, T, -1)

        # Forward.
        dist_flat = self.actor(c_t.reshape(B * T, -1))  # single B*T batch
        values = self.critic(c_t.reshape(B * T, -1)).reshape(B, T)

        with torch.no_grad():
            target_values = self.target_critic(
                c_t.reshape(B * T, -1)
            ).reshape(B, T)

        # Advantage (TD residual).
        advantages = compute_td_advantage(
            values, rewards, target_values, terminated, continuation,
            bootstrap_value, gamma,
        )

        # Losses — detach actions so mode()-sampled actions don't create
        # a dual gradient path through the same actor parameters.
        actor_loss = self.compute_actor_loss(
            dist_flat, actions.reshape(B * T, -1).detach(),
            advantages.reshape(B * T).detach(),
            entropy_coef=entropy_coef,
        )
        critic_loss = self.compute_critic_loss(
            values, target_values, rewards, terminated, continuation,
            bootstrap_value, gamma=gamma, lambda_=lambda_,
        )

        # Step.
        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()

        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()

        self.update_target()

        with torch.no_grad():
            entropy_val = dist_flat.entropy_sample().mean().item()
            value_mean = values.mean().item()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy_val,
            "value_mean": value_mean,
        }


# ---------------------------------------------------------------------------
# Return / advantage computation
# ---------------------------------------------------------------------------

def compute_lambda_returns(
    rewards: Tensor,               # (B, T)
    bootstrap_values: Tensor,      # (B, T) V(s_0)...V(s_{T-1})
    terminated: Tensor,            # (B, T) bool
    continuation: Tensor,          # (B, T) bool
    bootstrap_value: Tensor,       # (B,)   V_target(s_T)
    gamma: float = 0.997,
    lambda_: float = 0.95,
) -> Tensor:
    """λ-return for each timestep ``(B, T)``.

    Contract (see module docstring for details)::

        G_t = r_t + γ * continuation_t *
              ((1 - λ) * V_target(s_{t+1}) + λ * G_{t+1})

    where ``V_target(s_{t+1}) = bootstrap_values[:, t+1]`` for ``t < T-1``,
    and ``V_target(s_T) = bootstrap_value`` for the final-step bootstrap.

    Parameters
    ----------
    rewards:
        ``(B, T)``.
    bootstrap_values:
        ``(B, T)`` — ``V_target(s_0…s_{T-1})``.
    terminated:
        ``(B, T)`` bool — true terminal state (must imply
        ``continuation=False``).
    continuation:
        ``(B, T)`` bool — whether a valid next state exists for bootstrap.
    bootstrap_value:
        ``(B,)`` — ``V_target(s_T)``, used only when
        ``continuation[:, -1]`` is True.
    """
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    _validate_return_inputs(
        rewards, bootstrap_values, terminated, continuation, bootstrap_value,
    )

    uses_bootstrap = continuation.float() * gamma  # (B, T)

    returns = torch.zeros(B, T, device=device, dtype=dtype)

    # Last step: bootstrap from bootstrap_value if continuation, else r.
    returns[:, -1] = rewards[:, -1] + uses_bootstrap[:, -1] * bootstrap_value

    for t in range(T - 2, -1, -1):
        v_next = bootstrap_values[:, t + 1]  # V_target(s_{t+1})
        g_next = returns[:, t + 1]
        bootstrap = ((1.0 - lambda_) * v_next + lambda_ * g_next)
        returns[:, t] = rewards[:, t] + uses_bootstrap[:, t] * bootstrap

    return returns


def compute_td_advantage(
    values: Tensor,               # (B, T)  online values V(s_0…s_{T-1})
    rewards: Tensor,              # (B, T)
    bootstrap_values: Tensor,     # (B, T)  target V(s_0…s_{T-1})
    terminated: Tensor,           # (B, T)  bool
    continuation: Tensor,         # (B, T)  bool
    bootstrap_value: Tensor,      # (B,)    V_target(s_T)
    gamma: float = 0.997,
) -> Tensor:
    """TD(λ=0) advantage ``(B, T)``.

    ``A_t = r_t + γ * continuation_t * V_target(s_{t+1}) - V(s_t)``
    """
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    _validate_return_inputs(
        rewards, bootstrap_values, terminated, continuation, bootstrap_value,
    )

    # Build V_target(s_{t+1}) for t=0…T-1, with V(s_T) = bootstrap_value.
    next_v = torch.zeros(B, T, device=device, dtype=dtype)
    if T > 1:
        next_v[:, :-1] = bootstrap_values[:, 1:]
    next_v[:, -1] = bootstrap_value  # used only when continuation[:, -1] is True

    uses_bootstrap = continuation.float() * gamma
    adv = rewards + uses_bootstrap * next_v - values
    return adv
