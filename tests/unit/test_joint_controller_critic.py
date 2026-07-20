"""Stage 6.1 bounded ControllerTrunk/Critic training tests."""

from __future__ import annotations

import torch

from rwm.config.experiment_config import TemporalConfig
from rwm.models.actor_critic import ActorCritic
from rwm.models.rwm.model import ReducedWorldModel
from rwm.trainers.joint_controller_critic import (
    JointControllerCriticConfig,
    JointControllerCriticTrainer,
)


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    total = 36
    loss_mask = torch.zeros(batch_size, total, dtype=torch.bool)
    loss_mask[:, 20:] = True
    return {
        "obs": torch.randn(batch_size, total, 3, 64, 64),
        "action": torch.randn(batch_size, total, 3),
        "reward": torch.randn(batch_size, total),
        "valid_step": torch.ones(batch_size, total, dtype=torch.bool),
        "loss_mask": loss_mask,
        "predecessor_action": torch.randn(batch_size, 3),
    }


def _trainer() -> JointControllerCriticTrainer:
    model = ReducedWorldModel(
        temporal_config=TemporalConfig(backend="minimal_sru"),
        tokenizer_eval_mode="mean",
    )
    ac = ActorCritic(model)
    return JointControllerCriticTrainer(model, ac, device=torch.device("cpu"))


def test_declared_freeze_boundary() -> None:
    trainer = _trainer()
    assert all(p.requires_grad for p in trainer.model.controller.parameters())
    assert all(p.requires_grad for p in trainer.ac.critic.parameters())
    assert all(not p.requires_grad for p in trainer.ac.actor.parameters())
    assert all(not p.requires_grad for p in trainer.ac.target_critic.parameters())
    assert all(not p.requires_grad for p in trainer.model.world_hd.parameters())
    assert all(not p.requires_grad for p in trainer.model.encoder.parameters())


def test_optimizer_contains_only_controller_and_online_critic() -> None:
    trainer = _trainer()
    actual = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    expected = {
        id(parameter)
        for module in (trainer.model.controller, trainer.ac.critic)
        for parameter in module.parameters()
    }
    assert actual == expected


def test_one_step_changes_only_open_boundary() -> None:
    trainer = _trainer()
    controller_before = {
        name: value.clone() for name, value in trainer.model.controller.state_dict().items()
    }
    critic_before = {
        name: value.clone() for name, value in trainer.ac.critic.state_dict().items()
    }
    actor_before = {
        name: value.clone() for name, value in trainer.ac.actor.state_dict().items()
    }
    sru_before = {
        name: value.clone() for name, value in trainer.model.world_hd.state_dict().items()
    }
    metrics = trainer.train_step(_batch())
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(
        not torch.equal(value, trainer.model.controller.state_dict()[name])
        for name, value in controller_before.items()
    )
    assert any(
        not torch.equal(value, trainer.ac.critic.state_dict()[name])
        for name, value in critic_before.items()
    )
    assert all(
        torch.equal(value, trainer.ac.actor.state_dict()[name])
        for name, value in actor_before.items()
    )
    assert all(
        torch.equal(value, trainer.model.world_hd.state_dict()[name])
        for name, value in sru_before.items()
    )


def test_fixed_mask_scores_exact_horizon_per_sample() -> None:
    trainer = _trainer()
    batch = _batch()
    keep, scored = trainer._fixed_mask(batch["loss_mask"], batch["valid_step"])
    assert scored.sum().item() == 2 * trainer.cfg.horizon
    assert torch.equal(~keep, scored)


def test_config_rejects_invalid_values() -> None:
    for kwargs in (
        {"horizon": 0},
        {"warmup_steps": 0},
        {"critic_weight": 0.0},
        {"controller_lr": 0.0},
    ):
        try:
            JointControllerCriticConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"configuration should fail: {kwargs}")
