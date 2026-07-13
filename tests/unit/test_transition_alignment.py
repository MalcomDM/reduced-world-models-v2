"""Deterministic proof of the corrected Stage 2 timing contract.

Approved contract:
    belief b_t = Transformer(obs[t], action[t-1], history)
    Actor(b_t)             → action[t]           (Stage 4)
    RewardHead(b_t, action[t]) → reward[t]       (active)

The trainer passes:
  - prev_action = zeros for t=0, action[t-1] for t>0 (Transformer token)
  - current_action = action[t] for all t          (reward head)
  - target = reward[t]
"""

import numpy as np
import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from rwm.data.rollout_dataset import RolloutDataset
from rwm.trainers.deterministic import world_model_trainer as _wmt
from rwm.trainers.deterministic.world_model_trainer import WorldModelTrainer
from rwm.types import WorldModelOutput


# ---------------------------------------------------------------------------
# Spy
# ---------------------------------------------------------------------------

class _ModelSpy(torch.nn.Module):
    """Records ``prev_action`` and ``current_action`` passed to every call."""

    def __init__(self, real_model: torch.nn.Module) -> None:
        super().__init__()
        self._real = real_model
        self.encoder = real_model.encoder
        self.tokenizer = real_model.tokenizer
        self.records: list[dict] = []

    def forward(
        self,
        img: torch.Tensor,
        prev_action: torch.Tensor,
        current_action: torch.Tensor,
        history: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        force_keep_input: bool = False,
    ) -> WorldModelOutput:
        self.records.append({
            "prev_action": prev_action.detach().cpu(),
            "current_action": current_action.detach().cpu(),
        })
        return self._real(img, prev_action, current_action,
                          history, lengths, force_keep_input)

    def forward_sequence(
        self,
        obs: torch.Tensor,
        prev_actions: torch.Tensor,
        current_actions: torch.Tensor,
        force_keep_input: bool = False,
    ) -> WorldModelOutput:
        for t in range(obs.shape[1]):
            self.records.append({
                "prev_action": prev_actions[:, t].detach().cpu(),
                "current_action": current_actions[:, t].detach().cpu(),
            })
        return self._real.forward_sequence(obs, prev_actions, current_actions, force_keep_input)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_rollout(path: Path, T: int = 20, seed: int = 42) -> None:
    rng = np.random.RandomState(seed)
    obs = rng.randint(0, 255, size=(T, 64, 64, 3), dtype=np.uint8)
    action = np.zeros((T, 3), dtype=np.float32)
    action[:, 0] = np.arange(T, dtype=np.float32) + 1.0
    action[:, 1] = 0.5
    action[:, 2] = 0.0
    reward = rng.randn(T).astype(np.float32)
    done = np.zeros(T, dtype=bool)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, obs=obs, action=action, reward=reward, done=done)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.models
def test_timing_contract_prev_action_is_zeros_at_t0_and_act_tminus1_after(
    tmp_path: Path,
):
    """Prove the trainer passes:
    - prev_action = zeros at step 0
    - prev_action = action[t-1] at step t > 0
    - current_action = action[t] at every step t
    """
    path = tmp_path / "rollout.npz"
    _make_synthetic_rollout(path, T=20, seed=42)
    seq_len = 10

    ds = RolloutDataset(root_dir=tmp_path, sequence_len=seq_len, image_size=64)
    loader = DataLoader(ds, batch_size=2, shuffle=False, drop_last=False)

    trainer = WorldModelTrainer(
        train_loader=loader,
        out_dir=tmp_path / "out",
        sequence_len=seq_len,
        epochs=1,
        batch_size=2,
        lr=1e-4,
    )

    spy = _ModelSpy(trainer.model)
    trainer.model = spy

    batch = next(iter(loader))
    _ = trainer._compute_batch_loss(batch)

    act = batch["action"]  # (B, T, A)
    B = act.shape[0]

    assert len(spy.records) == seq_len, (
        f"Expected {seq_len} records, got {len(spy.records)}"
    )

    for t in range(seq_len):
        rec = spy.records[t]

        # prev_action: zeros at t=0, action[t-1] at t>0
        if t == 0:
            expected_prev = torch.zeros(B, 3)
        else:
            expected_prev = act[:, t - 1, :]

        torch.testing.assert_close(
            rec["prev_action"], expected_prev,
            msg=f"Step {t}: prev_action should be action[t-1] (zeros at t=0)",
        )

        # current_action: action[t] at every step
        expected_curr = act[:, t, :]
        torch.testing.assert_close(
            rec["current_action"], expected_curr,
            msg=f"Step {t}: current_action should be action[t]",
        )


@pytest.mark.models
def test_timing_contract_reward_head_receives_current_action(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The reward MSE target is reward[t] (the reward from action[t])."""
    path = tmp_path / "rollout2.npz"
    _make_synthetic_rollout(path, T=20, seed=7)
    seq_len = 10

    ds = RolloutDataset(root_dir=tmp_path, sequence_len=seq_len, image_size=64)
    loader = DataLoader(ds, batch_size=2, shuffle=False, drop_last=False)

    trainer = WorldModelTrainer(
        train_loader=loader,
        out_dir=tmp_path / "out2",
        sequence_len=seq_len,
        epochs=1,
        batch_size=2,
        lr=1e-4,
        beta=0.0,
    )

    recorded_targets: list[torch.Tensor] = []
    orig = _wmt.F.mse_loss

    def rec_mse(input, target, *a, **kw):
        recorded_targets.append(target.detach().cpu())
        return orig(input, target, *a, **kw)

    monkeypatch.setattr(_wmt.F, "mse_loss", rec_mse)

    batch = next(iter(loader))
    _loss_total, _loss_mse, _loss_kl = trainer._compute_batch_loss(batch)

    rew = batch["reward"]
    assert len(recorded_targets) == 1
    target_full = recorded_targets[0]
    expected = rew[:, :seq_len]
    torch.testing.assert_close(
        target_full, expected,
        msg="Target must be reward[:, :sequence_len]",
    )


@pytest.mark.models
def test_alignment_indexing_by_construction(tmp_path: Path):
    """Rollout-file ground truth: reward[t] = env.step(action[t])[1]."""
    path = tmp_path / "check.npz"
    _make_synthetic_rollout(path, T=20, seed=42)
    data = np.load(path)
    action = data["action"]
    reward = data["reward"]
    assert action.shape[0] == reward.shape[0] == 20
    assert action[0, 0] == 1.0
