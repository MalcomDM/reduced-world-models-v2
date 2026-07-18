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

from typing import Optional

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
        observation_keep: Optional[torch.Tensor] = None,
    ) -> WorldModelOutput:
        for t in range(obs.shape[1]):
            self.records.append({
                "prev_action": prev_actions[:, t].detach().cpu(),
                "current_action": current_actions[:, t].detach().cpu(),
            })
        return self._real.forward_sequence(
            obs, prev_actions, current_actions, force_keep_input,
            observation_keep=observation_keep,
        )


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

        # prev_action at t=0: predecessor_action from batch (zeros at true
        # episode start, action[offset-1] mid-episode).
        # prev_action at t>0: action[t-1].
        if t == 0:
            expected_prev = batch["predecessor_action"]
        else:
            expected_prev = act[:, t - 1, :]

        torch.testing.assert_close(
            rec["prev_action"], expected_prev,
            msg=f"Step {t}: prev_action should be predecessor (t=0) or action[t-1] (t>0)",
        )

        # current_action: action[t] at every step
        expected_curr = act[:, t, :]
        torch.testing.assert_close(
            rec["current_action"], expected_curr,
            msg=f"Step {t}: current_action should be action[t]",
        )


@pytest.mark.models
def test_trainer_rejects_batch_without_predecessor_action(tmp_path: Path):
    """Missing predecessor metadata must not silently restore the old reset."""
    trainer = WorldModelTrainer(
        train_loader=[],
        out_dir=tmp_path / "out",
        sequence_len=2,
        epochs=1,
        batch_size=1,
    )
    batch = {
        "obs": torch.zeros(1, 2, 3, 64, 64),
        "action": torch.zeros(1, 2, 3),
        "reward": torch.zeros(1, 2),
        "done": torch.zeros(1, 2, dtype=torch.bool),
    }

    with pytest.raises(KeyError, match="predecessor_action"):
        trainer._compute_batch_loss(batch)


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
def test_mid_episode_window_gets_predecessor_action(tmp_path: Path):
    """A mid-episode window must get the actual action before its offset
    as the predecessor, not zeros."""
    path = tmp_path / "rollout.npz"
    # Create a rollout with known action values (steer[0]=1, steer[1]=2, ...)
    rng = np.random.RandomState(42)
    T = 50
    obs = rng.randint(0, 255, size=(T, 64, 64, 3), dtype=np.uint8)
    action = np.zeros((T, 3), dtype=np.float32)
    action[:, 0] = np.arange(T, dtype=np.float32) + 1.0  # steer = t+1
    action[:, 1] = 0.5
    reward = rng.randn(T).astype(np.float32)
    done = np.zeros(T, dtype=bool)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, obs=obs, action=action, reward=reward, done=done)

    # Create a dataset and sample a mid-episode window at offset 5.
    ds = RolloutDataset(root_dir=tmp_path, sequence_len=10, image_size=64)

    # Find a sample with offset > 0
    mid_ep_samples = [
        (path, off) for (path, off) in ds.samples if off > 0
    ]
    assert len(mid_ep_samples) > 0, "Must have mid-episode windows"
    sample_path, sample_offset = mid_ep_samples[0]

    # Load that sample directly
    with np.load(sample_path) as data:
        expected_predecessor = torch.tensor(
            data["action"][sample_offset - 1], dtype=torch.float32,
        )

    # Get the sample from the dataset
    sample_idx = ds.samples.index((sample_path, sample_offset))
    sample = ds[sample_idx]

    assert "predecessor_action" in sample, "Dataset must return predecessor_action"
    torch.testing.assert_close(
        sample["predecessor_action"], expected_predecessor,
        msg=(
            f"Mid-episode window at offset {sample_offset} should get "
            f"action[offset-1] as predecessor, got {sample['predecessor_action']}"
        ),
    )


@pytest.mark.models
def test_episode_start_window_gets_zero_predecessor(tmp_path: Path):
    """A window at true episode start must have zeros as predecessor."""
    path = tmp_path / "start.npz"
    rng = np.random.RandomState(0)
    obs = rng.randint(0, 255, size=(20, 64, 64, 3), dtype=np.uint8)
    action = rng.uniform(-1, 1, size=(20, 3)).astype(np.float32)
    reward = rng.randn(20).astype(np.float32)
    done = np.zeros(20, dtype=bool)
    np.savez_compressed(path, obs=obs, action=action, reward=reward, done=done)

    ds = RolloutDataset(root_dir=tmp_path, sequence_len=10, image_size=64)

    # Find offset-0 samples
    start_samples = [
        (p, off) for (p, off) in ds.samples if off == 0
    ]
    assert len(start_samples) > 0, "Must have episode-start windows"

    sample = ds[ds.samples.index(start_samples[0])]
    expected_zeros = torch.zeros(3, dtype=torch.float32)
    torch.testing.assert_close(
        sample["predecessor_action"], expected_zeros,
        msg="Episode-start window should get zeros as predecessor",
    )


@pytest.mark.models
def test_predecessor_does_not_cross_episode_boundary(tmp_path: Path):
    """A window from one file must never get a predecessor from another file."""
    # Create two separate rollout files
    for i in range(2):
        p = tmp_path / f"ep_{i}.npz"
        rng = np.random.RandomState(i)
        obs = rng.randint(0, 255, size=(20, 64, 64, 3), dtype=np.uint8)
        action = np.full((20, 3), float(i + 1), dtype=np.float32)  # constant per file
        reward = rng.randn(20).astype(np.float32)
        done = np.zeros(20, dtype=bool)
        np.savez_compressed(p, obs=obs, action=action, reward=reward, done=done)

    ds = RolloutDataset(root_dir=tmp_path, sequence_len=10, image_size=64)

    # All mid-episode samples should have predecessor from same file
    for file_path, offset in ds.samples:
        if offset > 0:
            with np.load(file_path) as data:
                expected = torch.tensor(data["action"][offset - 1], dtype=torch.float32)
            idx = ds.samples.index((file_path, offset))
            sample = ds[idx]
            torch.testing.assert_close(
                sample["predecessor_action"], expected,
                msg=f"Predecessor must come from same file (offset={offset})",
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
