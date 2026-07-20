"""Regression tests for the no-retrain checkpoint evaluator."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader


_SCRIPT = Path(__file__).parents[2] / "scripts" / "evaluation" / "evaluate_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_checkpoint", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
evaluate_loader = _MODULE.evaluate_loader
reward_mean = _MODULE.reward_mean


class _SequenceSpy:
    def __init__(self) -> None:
        self.prev_actions = None
        self.calls = 0

    def eval(self):
        return self

    def forward_sequence(self, obs, prev_actions, actions, force_keep_input=True, **kwargs):
        self.calls += 1
        self.prev_actions = prev_actions.detach().clone()
        return SimpleNamespace(reward_pred_seq=torch.zeros_like(actions[..., 0]))


def _batch(predecessor, actions, rewards):
    sequence_len = actions.shape[0]
    return {
        "obs": torch.zeros(sequence_len, 3, 64, 64),
        "action": actions,
        "reward": rewards,
        "predecessor_action": predecessor,
    }


def test_evaluator_uses_sequence_contract_and_counts_all_transitions():
    actions = torch.tensor([[
        [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
    ]])
    predecessor = torch.tensor([[9.0, 0.0, 0.0]])
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    loader = DataLoader([_batch(predecessor[0], actions[0], rewards[0])], batch_size=1)
    spy = _SequenceSpy()

    metrics = evaluate_loader(spy, loader, torch.device("cpu"), train_reward_mean=2.0)

    assert spy.calls == 1
    torch.testing.assert_close(
        spy.prev_actions,
        torch.tensor([[[9.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
    )
    assert metrics["transitions"] == 3
    assert metrics["val_mse"] == pytest.approx(14.0 / 3.0)
    assert metrics["val_mae"] == pytest.approx(2.0)
    assert metrics["baseline_mse"] == pytest.approx(2.0 / 3.0)


def test_evaluator_counts_every_item_when_batch_has_no_masks():
    """A broadcast fallback mask must still count all B×T transitions."""
    first = _batch(
        torch.zeros(3),
        torch.zeros(3, 3),
        torch.tensor([1.0, 2.0, 3.0]),
    )
    second = _batch(
        torch.zeros(3),
        torch.zeros(3, 3),
        torch.tensor([4.0, 5.0, 6.0]),
    )
    loader = DataLoader([first, second], batch_size=2)

    metrics = evaluate_loader(
        _SequenceSpy(), loader, torch.device("cpu"), train_reward_mean=3.5,
    )

    assert metrics["transitions"] == 6
    assert metrics["val_mse"] == pytest.approx(91.0 / 6.0)


def test_reward_mean_uses_all_training_transitions_not_batch_means():
    first = _batch(torch.zeros(3), torch.zeros(2, 3), torch.tensor([1.0, 1.0]))
    second = _batch(torch.zeros(3), torch.zeros(1, 3), torch.tensor([10.0]))
    loader = DataLoader([first, second], batch_size=1)

    assert reward_mean(loader) == pytest.approx(4.0)


def test_tail_16_evaluator_and_baseline_use_only_final_positions():
    """The actual evaluator and constant baseline share the tail-16 mask."""
    rewards = torch.cat((torch.full((20,), 100.0), torch.arange(16, dtype=torch.float32)))
    sample = {
        "obs": torch.zeros(36, 3, 64, 64),
        "action": torch.zeros(36, 3),
        "reward": rewards,
        "predecessor_action": torch.zeros(3),
        "valid_step": torch.ones(36, dtype=torch.bool),
        "loss_mask": torch.ones(36, dtype=torch.bool),
    }
    loader = DataLoader([sample], batch_size=1)
    tail_mean = reward_mean(loader, eval_mode="tail_16")

    metrics = evaluate_loader(
        _SequenceSpy(), loader, torch.device("cpu"),
        train_reward_mean=tail_mean, eval_mode="tail_16",
    )

    assert tail_mean == pytest.approx(7.5)
    assert metrics["transitions"] == 16
    assert metrics["val_mse"] == pytest.approx(
        torch.arange(16, dtype=torch.float32).square().mean().item()
    )
    assert metrics["baseline_mse"] == pytest.approx(
        (torch.arange(16, dtype=torch.float32) - 7.5).square().mean().item()
    )


def test_sru_context_evaluation_scores_only_target_positions():
    """Frozen SRU evaluation must rebuild burn-in and exclude it from metrics."""
    sample = {
        "obs": torch.zeros(4, 3, 64, 64),
        "action": torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                                [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        "reward": torch.tensor([99.0, 99.0, 1.0, 3.0]),
        "predecessor_action": torch.tensor([7.0, 0.0, 0.0]),
        "valid_step": torch.tensor([False, True, True, True]),
        "loss_mask": torch.tensor([False, False, True, True]),
    }
    loader = DataLoader([sample], batch_size=1)
    spy = _SequenceSpy()

    metrics = evaluate_loader(spy, loader, torch.device("cpu"), train_reward_mean=2.0)

    assert metrics["transitions"] == 2
    assert metrics["val_mse"] == pytest.approx(5.0)
    assert metrics["baseline_mse"] == pytest.approx(1.0)
    # The first real position is index 1, so its predecessor is restored there.
    assert spy.prev_actions[0, 1, 0].item() == pytest.approx(7.0)
    assert reward_mean(loader) == pytest.approx(2.0)
