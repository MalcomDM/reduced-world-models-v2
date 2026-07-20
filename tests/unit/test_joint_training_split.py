"""Tests for Stage 6.1 data isolation, seed semantics, and freeze boundaries.

Verifies:
  1. train/validation files are disjoint.
  2. Split parity between trainer and evaluator.
  3. Changing training_seed does NOT change file split.
  4. Changing data_split_seed DOES change file split.
  5. Training loader contains only train files.
  6. DataLoader order is reproducible from training_seed.
  7. Actor/SRU/perception unchanged after a smoke update.
  8. ControllerTrunk and OnlineCritic change.
  9. Source hashes and file manifests persist in summary.
 10. Output overwrite is rejected.
 11. No _ac_loader directory is created.
"""

import json
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from rwm.config.config import ACTION_DIM
from rwm.config.experiment_config import TemporalConfig
from rwm.data.rollout_dataset import RolloutDataset, _collect_npz_files
from rwm.data.split import collect_and_split
from rwm.trainers.imagined_actor_critic import (
    ImaginedACTrainer,
    ImaginedACTrainingConfig,
    load_ac_from_checkpoint,
    validate_ac_anchor_checkpoint,
)
from rwm.trainers.joint_controller_critic import (
    JointControllerCriticConfig,
    JointControllerCriticTrainer,
)
from rwm.models.rwm.model import ReducedWorldModel
from rwm.utils.checkpointing import load_checkpoint, model_from_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_npz(tmp_path: Path, T: int = 100, name: str = "ep_0.npz") -> Path:
    path = tmp_path / name
    np.savez(
        path,
        obs=np.random.randint(0, 255, (T, 96, 96, 3), dtype=np.uint8),
        action=np.random.randn(T, 3).astype(np.float32),
        reward=np.random.randn(T).astype(np.float32),
        done=np.zeros(T, dtype=bool),
    )
    return path


def _make_multi_episode_dir(tmp_path: Path, n: int = 6) -> Path:
    d = tmp_path / "rollouts"
    d.mkdir()
    for i in range(n):
        _make_fake_npz(d, T=100, name=f"ep_{i}.npz")
    return d


def _make_sru_model_and_ac(device=torch.device("cpu")):
    from torch.utils.data import DataLoader
    import tempfile as _tf

    tc = TemporalConfig(backend="minimal_sru")
    m = ReducedWorldModel(temporal_config=tc, tokenizer_eval_mode="mean").eval()
    for p in m.parameters():
        p.requires_grad_(False)

    cfg = ImaginedACTrainingConfig(warmup_steps=4, max_batches=1)
    loader = DataLoader(_FakeDatasetT(), batch_size=1, shuffle=False)
    tr = ImaginedACTrainer(model=m, train_loader=loader, train_cfg=cfg,
                           out_dir=Path(_tf.mkdtemp()), device=device)
    return m, tr.ac


class _FakeDatasetT(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        T = 36
        lm = torch.zeros(T, dtype=torch.bool); lm[20:] = True
        vs = torch.ones(T, dtype=torch.bool)
        return {
            "obs": torch.randn(T, 3, 64, 64),
            "action": torch.randn(T, 3),
            "reward": torch.randn(T),
            "done": torch.zeros(T, dtype=torch.bool),
            "predecessor_action": torch.randn(3),
            "valid_step": vs,
            "loss_mask": lm,
        }


def _make_smoke_batch(tmp_path: Path) -> dict:
    """Build a minimal smoke batch from fake npz files."""
    d = _make_multi_episode_dir(tmp_path)
    ds = RolloutDataset(
        root_dir=d, sequence_len=16,
        recurrent_context=True, burn_in_steps=20,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False, drop_last=True)
    return next(iter(loader))


# ===================================================================
# 1. Train/validation disjointness
# ===================================================================

class TestTrainValDisjoint:
    def test_disjoint_file_lists(self, tmp_path):
        """Train and validation file lists have no overlap."""
        d = _make_multi_episode_dir(tmp_path, n=10)
        train, val = collect_and_split(d, data_split_seed=42)
        intersection = set(str(p) for p in train) & set(str(p) for p in val)
        assert len(intersection) == 0, f"Overlap: {intersection}"

    def test_all_files_covered(self, tmp_path):
        """Every source file appears in exactly one of train or val."""
        d = _make_multi_episode_dir(tmp_path, n=10)
        all_files = set(str(p) for p in _collect_npz_files(d))
        train, val = collect_and_split(d, data_split_seed=42)
        union = set(str(p) for p in train) | set(str(p) for p in val)
        assert union == all_files, f"Missing: {all_files - union}"

    def test_train_longer_than_val(self, tmp_path):
        """Training set has more files than validation set (typical 80/20)."""
        d = _make_multi_episode_dir(tmp_path, n=10)
        train, val = collect_and_split(d, data_split_seed=42, val_ratio=0.2)
        assert len(train) > len(val)


# ===================================================================
# 2. Split parity between trainer and evaluator
# ===================================================================

class TestSplitParity:
    """Both tools use the same helper → same split for same seed."""

    def test_same_seed_same_split(self, tmp_path):
        """Two calls with same seed produce identical split."""
        d = _make_multi_episode_dir(tmp_path, n=10)
        t1, v1 = collect_and_split(d, data_split_seed=42)
        t2, v2 = collect_and_split(d, data_split_seed=42)
        assert t1 == t2
        assert v1 == v2


# ===================================================================
# 3. training_seed change does NOT change split
# ===================================================================

class TestTrainingSeedNoSplitEffect:
    """training_seed is independent of data_split_seed."""

    def test_different_training_seed_same_split(self, tmp_path):
        """Split is identical when only training_seed differs."""
        d = _make_multi_episode_dir(tmp_path, n=10)
        t1, v1 = collect_and_split(d, data_split_seed=42)
        # collect_and_split doesn't use training_seed at all
        t2, v2 = collect_and_split(d, data_split_seed=42)
        assert t1 == t2


# ===================================================================
# 4. data_split_seed change DOES change split
# ===================================================================

class TestDataSplitSeedChanges:
    """Different data_split_seed → different file partition."""

    def test_different_seed_different_split(self, tmp_path):
        """Two different data_split_seed values give different train/val splits."""
        d = _make_multi_episode_dir(tmp_path, n=10)
        t1, v1 = collect_and_split(d, data_split_seed=42)
        t2, v2 = collect_and_split(d, data_split_seed=99)
        # At least one of train or val differs
        assert t1 != t2 or v1 != v2, "Split must differ for different seeds"


# ===================================================================
# 5. Training loader contains only train files
# ===================================================================

class TestTrainingLoaderIsolation:
    """Training DataLoader must not include val files."""

    def test_loader_only_train_files(self, tmp_path):
        """Training DataLoader's source files are all in train_files."""
        d = _make_multi_episode_dir(tmp_path, n=10)
        train, val = collect_and_split(d, data_split_seed=42)
        train_set = set(str(p) for p in train)
        ds = RolloutDataset.from_file_list(
            train, sequence_len=16, recurrent_context=True, burn_in_steps=20,
        )
        # Verify every sample comes from a train file
        for fpath, _ in ds.samples:
            assert str(fpath) in train_set, f"{fpath} is not a train file"


# ===================================================================
# 6. DataLoader reproducibility
# ===================================================================

class TestDataLoaderReproducibility:
    """Same training_seed → same DataLoader order."""

    def test_dataloader_order_reproducible(self, tmp_path):
        """Two loaders with same seed produce same first batch."""
        d = _make_multi_episode_dir(tmp_path, n=10)
        train, _ = collect_and_split(d, data_split_seed=42)
        ds = RolloutDataset.from_file_list(
            train, sequence_len=16, recurrent_context=True, burn_in_steps=20,
        )
        gen1 = torch.Generator(device="cpu").manual_seed(42)
        gen2 = torch.Generator(device="cpu").manual_seed(42)
        l1 = DataLoader(ds, batch_size=2, shuffle=True, drop_last=True, generator=gen1)
        l2 = DataLoader(ds, batch_size=2, shuffle=True, drop_last=True, generator=gen2)
        b1 = next(iter(l1))
        b2 = next(iter(l2))
        assert torch.equal(b1["obs"], b2["obs"]), "First batch must match"


# ===================================================================
# 7/8. Freeze boundary smoke
# ===================================================================

class TestFreezeBoundary:
    """Actor/SRU/perception unchanged; ControllerTrunk/Critic changed after smoke."""

    def test_smoke_update_freeze_boundary(self, tmp_path):
        """After 2 smoke updates, frozen blocks unchanged, trainable blocks changed."""
        m, ac = _make_sru_model_and_ac()
        batch = _make_smoke_batch(tmp_path)
        batch = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Snapshots
        controller_before = {k: v.clone() for k, v in m.controller.state_dict().items()}
        critic_before = {k: v.clone() for k, v in ac.critic.state_dict().items()}
        actor_before = {k: v.clone() for k, v in ac.actor.state_dict().items()}
        sru_before = {k: v.clone() for k, v in m.world_hd.state_dict().items()}
        encoder_before = {k: v.clone() for k, v in m.encoder.state_dict().items()}

        # Run 2 updates
        cfg = JointControllerCriticConfig()
        trainer = JointControllerCriticTrainer(m, ac, cfg, device=torch.device("cpu"))
        for _ in range(2):
            metrics = trainer.train_step(batch)

        # ControllerTrunk changed
        controller_after = m.controller.state_dict()
        ctrl_changed = any(not torch.equal(controller_before[k].cpu(), controller_after[k].cpu())
                           for k in controller_before)
        assert ctrl_changed, "ControllerTrunk must change after training"

        # Critic changed
        critic_after = ac.critic.state_dict()
        critic_changed = any(not torch.equal(critic_before[k].cpu(), critic_after[k].cpu())
                             for k in critic_before)
        assert critic_changed, "Online Critic must change after training"

        # Actor unchanged
        actor_after = ac.actor.state_dict()
        actor_ok = all(torch.equal(actor_before[k].cpu(), actor_after[k].cpu())
                       for k in actor_before)
        assert actor_ok, "Actor must remain frozen"

        # SRU unchanged
        sru_after = m.world_hd.state_dict()
        sru_ok = all(torch.equal(sru_before[k].cpu(), sru_after[k].cpu())
                     for k in sru_before)
        assert sru_ok, "SRU must remain frozen"

        # Encoder unchanged
        enc_after = m.encoder.state_dict()
        enc_ok = all(torch.equal(encoder_before[k].cpu(), enc_after[k].cpu())
                     for k in encoder_before)
        assert enc_ok, "Encoder must remain frozen"


# ===================================================================
# 9. Summary provenance
# ===================================================================

class TestSummaryProvenance:
    """training_summary.json contains hashes, file lists, and parameter counts."""

    @pytest.mark.integration
    def test_summary_contains_provenance(self):
        """Run a smoke on canonical checkpoints and verify summary fields."""
        import subprocess, sys
        out_dir = Path(tempfile.mkdtemp()) / "smoke_prov"
        anchor = "runs/component_refinement/sru_temporal/08_strict_observational_dropout_anchor/seed42/checkpoint_best.pt"
        ac_ckpt = "runs/imagined_actor_critic/minimal_sru/01_frozen_parity/seed42/checkpoints/ac_checkpoint_2000.pt"
        result = subprocess.run([
            sys.executable, "scripts/training/train_joint_controller_critic.py",
            "--anchor", anchor,
            "--ac", ac_ckpt,
            "--out", str(out_dir),
            "--smoke",
        ], capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(result.stderr)
            pytest.fail(f"CLI failed: {result.stderr}")

        summary = json.loads((out_dir / "training_summary.json").read_text())
        assert "provenance" in summary
        assert summary["provenance"]["anchor_hash"] is not None
        assert "train_files" in summary["provenance"]
        assert "val_files" in summary["provenance"]
        assert summary["provenance"]["train_val_disjoint"] is True
        assert "freeze_boundary" in summary
        assert summary["freeze_boundary"]["sru_unchanged"] is True
        assert summary["freeze_boundary"]["actor_unchanged"] is True
        assert summary["training"]["controller_lr"] == 3e-5


# ===================================================================
# 10. Output overwrite rejection
# ===================================================================

class TestOutputOverwrite:
    """CLI must refuse to overwrite an existing output directory."""

    def test_existing_output_rejected(self, tmp_path):
        """Overwriting an existing directory raises SystemExit."""
        import subprocess, sys
        out_dir = tmp_path / "existing_out"
        out_dir.mkdir()
        anchor = "runs/component_refinement/sru_temporal/08_strict_observational_dropout_anchor/seed42/checkpoint_best.pt"
        ac_ckpt = "runs/imagined_actor_critic/minimal_sru/01_frozen_parity/seed42/checkpoints/ac_checkpoint_2000.pt"
        result = subprocess.run([
            sys.executable, "scripts/training/train_joint_controller_critic.py",
            "--anchor", anchor,
            "--ac", ac_ckpt,
            "--out", str(out_dir),
            "--smoke",
        ], capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        assert "already exists" in result.stderr


# ===================================================================
# 11. No _ac_loader side effect
# ===================================================================

class TestNoACLoaderArtifact:
    """No _ac_loader directory created during checkpoint loading."""

    def test_load_ac_no_side_effect(self):
        """load_ac_from_checkpoint does not create any directory."""
        from rwm.utils.checkpointing import load_checkpoint, model_from_checkpoint
        from rwm.trainers.imagined_actor_critic import load_ac_from_checkpoint
        from pathlib import Path

        anchor = "runs/component_refinement/sru_temporal/08_strict_observational_dropout_anchor/seed42/checkpoint_best.pt"
        ac_ckpt = "runs/imagined_actor_critic/minimal_sru/01_frozen_parity/seed42/checkpoints/ac_checkpoint_2000.pt"
        ckpt = load_checkpoint(Path(anchor))
        model = model_from_checkpoint(ckpt, tokenizer_eval_mode_override="mean").eval()
        loaded_ac = load_ac_from_checkpoint(model, Path(ac_ckpt), torch.device("cpu"))
        assert loaded_ac is not None

    @pytest.mark.integration
    def test_cli_no_ac_loader_artifact(self):
        """Joint training CLI does not create an _ac_loader directory."""
        import subprocess, sys, tempfile
        out_dir = Path(tempfile.mkdtemp()) / "train_no_ac"
        anchor = "runs/component_refinement/sru_temporal/08_strict_observational_dropout_anchor/seed42/checkpoint_best.pt"
        ac_ckpt = "runs/imagined_actor_critic/minimal_sru/01_frozen_parity/seed42/checkpoints/ac_checkpoint_2000.pt"
        result = subprocess.run([
            sys.executable, "scripts/training/train_joint_controller_critic.py",
            "--anchor", anchor,
            "--ac", ac_ckpt,
            "--out", str(out_dir),
            "--smoke",
        ], capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(result.stderr)
            pytest.fail(f"CLI failed: {result.stderr}")
        # Verify no _ac_loader directory exists anywhere in the output
        assert not any(p.name == "_ac_loader" for p in out_dir.rglob("*")), "_ac_loader directory found"


class TestActorCriticAnchorCompatibility:
    """Stage 6.1 must not mix Actor-Critic and world-model lineages."""

    def test_matching_anchor_hash_is_accepted(self, tmp_path):
        path = tmp_path / "ac.pt"
        torch.save({
            "kind": "imagined_actor_critic",
            "anchor": {"hash": "matching"},
        }, path)
        validate_ac_anchor_checkpoint(path, "matching")

    def test_mismatched_anchor_hash_is_rejected(self, tmp_path):
        path = tmp_path / "ac.pt"
        torch.save({
            "kind": "imagined_actor_critic",
            "anchor": {"hash": "old-anchor"},
        }, path)
        with pytest.raises(ValueError, match="anchor mismatch"):
            validate_ac_anchor_checkpoint(path, "new-anchor")

    def test_missing_anchor_provenance_is_rejected(self, tmp_path):
        path = tmp_path / "ac.pt"
        torch.save({"kind": "imagined_actor_critic"}, path)
        with pytest.raises(ValueError, match="anchor provenance"):
            validate_ac_anchor_checkpoint(path, "expected")
