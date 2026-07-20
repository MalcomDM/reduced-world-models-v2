"""Stage 6.1 controller/critic joint optimisation.

This is the first deliberately narrow end-to-end step.  Factual reward losses
and imagined Critic pressure may update the ControllerTrunk, while perception,
MinimalSRU, Actor, and TargetCritic remain frozen.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from rwm.evaluation.joint_gradient_audit import (
    _build_prev_actions,
    _extract_warmup_window,
)
from rwm.imagination import ImaginationRollout
from rwm.models.actor_critic import ActorCritic, compute_lambda_returns
from rwm.models.rwm.model import ReducedWorldModel


@dataclasses.dataclass(frozen=True)
class JointControllerCriticConfig:
    horizon: int = 4
    warmup_steps: int = 4
    gamma: float = 0.997
    lambda_: float = 0.95
    factual_weight: float = 1.0
    critic_weight: float = 0.5
    controller_lr: float = 3e-5
    critic_lr: float = 3e-4
    controller_grad_clip: float = 1.0
    critic_grad_clip: float = 5.0

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.warmup_steps < 1:
            raise ValueError("warmup_steps must be >= 1")
        for name in ("factual_weight", "critic_weight", "controller_lr", "critic_lr"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")


class JointControllerCriticTrainer:
    """Update only ControllerTrunk and OnlineCritic from reconstructed data."""

    def __init__(
        self,
        model: ReducedWorldModel,
        ac: ActorCritic,
        config: Optional[JointControllerCriticConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = config or JointControllerCriticConfig()
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.ac = ac.to(self.device)
        if self.model._temporal_backend != "minimal_sru":
            raise ValueError("Stage 6.1 requires temporal_backend='minimal_sru'")

        # Freeze everything, then open only the declared first-stage boundary.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.ac.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.controller.parameters():
            parameter.requires_grad_(True)
        for parameter in self.ac.critic.parameters():
            parameter.requires_grad_(True)

        # Keep frozen perception deterministic and prevent BatchNorm drift.
        self.model.eval()
        self.model.tokenizer.eval_mode = "mean"
        self.model.controller.train()
        self.ac.actor.eval()
        self.ac.critic.train()
        self.ac.target_critic.eval()

        self.imag = ImaginationRollout(self.model)
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": list(self.model.controller.parameters()),
                    "lr": self.cfg.controller_lr,
                    "name": "controller",
                },
                {
                    "params": list(self.ac.critic.parameters()),
                    "lr": self.cfg.critic_lr,
                    "name": "online_critic",
                },
            ]
        )

    def _fixed_mask(self, loss_mask: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor]:
        B, T = loss_mask.shape
        keep = torch.ones(B, T, dtype=torch.bool, device=loss_mask.device)
        scored = torch.zeros_like(keep)
        first_target = loss_mask.long().argmax(dim=1)
        for b in range(B):
            start = int(first_target[b].item()) + self.cfg.warmup_steps
            end = min(start + self.cfg.horizon, T)
            keep[b, start:end] = False
            scored[b, start:end] = True
        scored &= loss_mask & valid_step
        return keep, scored

    @staticmethod
    def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        count = mask.sum()
        if int(count.item()) == 0:
            raise ValueError("loss mask contains no supervised transitions")
        return ((prediction - target).square() * mask).sum() / count

    def _imagined_critic_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        valid_step: Tensor,
        loss_mask: Tensor,
        predecessor_action: Tensor,
    ) -> tuple[Tensor, Tensor]:
        warm = _extract_warmup_window(
            obs, actions, valid_step, loss_mask, predecessor_action,
            ws=self.cfg.warmup_steps,
        )
        # The lower world model is frozen in Stage 6.1.  Reconstruct the
        # current z from factual frames, but do not retain its graph.
        with torch.no_grad():
            state = self.imag.warmup(
                warm["obs"], warm["prev_actions"], warm["actions"],
                force_keep_input=True, valid_step=warm["valid_step"],
            )
            z_t = state.current_belief
            states = []
            rewards = []
            for _ in range(self.cfg.horizon):
                c_t = self.model.controller.encode(z_t)
                action = self.ac.actor(c_t).rsample().detach()
                _, reward = self.model.controller(z_t, action)
                states.append(z_t)
                rewards.append(reward.squeeze(-1))
                _, _, z_t = self.imag.advance(
                    state.history, state.lengths, action, temporal_state=z_t,
                )
            states_t = torch.stack(states, dim=1)
            rewards_t = torch.stack(rewards, dim=1)
            z_h = z_t

        B, H, D = states_t.shape
        shared = self.model.controller.encode(states_t.reshape(B * H, D))
        values = self.ac.critic(shared).reshape(B, H)
        with torch.no_grad():
            target_values = self.ac.target_critic(shared.detach()).reshape(B, H)
            bootstrap_shared = self.model.controller.encode(z_h)
            bootstrap = self.ac.target_critic(bootstrap_shared).squeeze(-1)
            terminated = torch.zeros(B, H, dtype=torch.bool, device=self.device)
            continuation = torch.ones(B, H, dtype=torch.bool, device=self.device)
            returns = compute_lambda_returns(
                rewards_t, target_values, terminated, continuation, bootstrap,
                self.cfg.gamma, self.cfg.lambda_,
            )
        return F.mse_loss(values, returns), rewards_t.mean()

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        obs = batch["obs"].to(self.device, non_blocking=True)
        actions = batch["action"].to(self.device, non_blocking=True)
        rewards = batch["reward"].to(self.device, non_blocking=True)
        valid = batch["valid_step"].to(self.device, non_blocking=True)
        loss_mask = batch["loss_mask"].to(self.device, non_blocking=True)
        predecessor = batch["predecessor_action"].to(self.device, non_blocking=True)
        prev_actions = _build_prev_actions(actions, valid, predecessor, obs.shape[1])

        visible = self.model.forward_sequence(
            obs, prev_actions, actions, force_keep_input=True,
            valid_step=valid, observation_dropout_execution="post_perception",
        )
        factual_mask = valid & loss_mask
        visible_mse = self._masked_mse(
            visible.reward_pred_seq, rewards, factual_mask,
        )

        keep, blind_mask = self._fixed_mask(loss_mask, valid)
        masked = self.model.forward_sequence(
            obs, prev_actions, actions, force_keep_input=True,
            observation_keep=keep, valid_step=valid,
            observation_dropout_execution="pre_perception_skip",
        )
        masked_mse = self._masked_mse(
            masked.reward_pred_seq, rewards, blind_mask,
        )
        critic_loss, imagined_reward_mean = self._imagined_critic_loss(
            obs, actions, valid, loss_mask, predecessor,
        )

        total = (
            self.cfg.factual_weight * (visible_mse + masked_mse)
            + self.cfg.critic_weight * critic_loss
        )
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        controller_grad = torch.nn.utils.clip_grad_norm_(
            self.model.controller.parameters(), self.cfg.controller_grad_clip,
        )
        critic_grad = torch.nn.utils.clip_grad_norm_(
            self.ac.critic.parameters(), self.cfg.critic_grad_clip,
        )
        self.optimizer.step()
        self.ac.update_target()

        return {
            "total_loss": float(total.detach().item()),
            "visible_mse": float(visible_mse.detach().item()),
            "masked_mse": float(masked_mse.detach().item()),
            "critic_loss": float(critic_loss.detach().item()),
            "imagined_reward_mean": float(imagined_reward_mean.detach().item()),
            "controller_grad_norm": float(controller_grad.detach().item()),
            "critic_grad_norm": float(critic_grad.detach().item()),
        }
