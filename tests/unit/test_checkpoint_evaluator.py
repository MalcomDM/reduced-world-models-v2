"""Regression tests for the no-retrain checkpoint evaluator."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader


_SCRIPT = Path(__file__).parents[2] / "scripts" / "evaluate_checkpoint.py"
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

    def forward_sequence(self, obs, prev_actions, actions, force_keep_input=True):
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


def test_reward_mean_uses_all_training_transitions_not_batch_means():
    first = _batch(torch.zeros(3), torch.zeros(2, 3), torch.tensor([1.0, 1.0]))
    second = _batch(torch.zeros(3), torch.zeros(1, 3), torch.tensor([10.0]))
    loader = DataLoader([first, second], batch_size=1)

    assert reward_mean(loader) == pytest.approx(4.0)
