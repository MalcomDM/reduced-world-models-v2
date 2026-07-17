"""Tests for Stage 0.5 experiment infrastructure.

Covers:
- Config JSON round-trip and deterministic serialization
- Run directory creation and artifact layout
- Dataset manifest creation, persistence, validation
- Structured checkpoint save/load
- Legacy bare state_dict checkpoint load
- Seed reproducibility
"""

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

# ---- Config ----
from rwm.config.experiment_config import (
    DataConfig,
    PerceptionConfig,
    TemporalConfig,
    ControllerConfig,
    TrainingConfig,
    ExperimentConfig,
)

# ---- Run directory ----
from rwm.utils.run_directory import (
    create_run_directory,
    find_run_dir,
    load_run_config,
)

# ---- Dataset manifest ----
from rwm.utils.dataset_manifest import (
    build_dataset_manifest,
    validate_manifest,
    save_manifest,
    load_manifest,
)

# ---- Checkpoint ----
from rwm.utils.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    _CHECKPOINT_SCHEMA_VERSION,
)

# ---- Seeding ----
from rwm.utils.seeding import (
    set_seed,
    SeedContext,
    get_current_seed,
    get_deterministic_flag,
)


# ===================================================================
# Config tests
# ===================================================================

class TestConfig:
    def test_defaults_match_current(self):
        """Default values should match the current ``config.py`` constants."""
        data = DataConfig()
        assert data.sequence_len == 16
        assert data.image_size == 64
        assert data.num_workers == 6

        perception = PerceptionConfig()
        assert perception.conv_filters == [32, 64, 16]
        assert perception.token_dim == 16
        assert perception.k == 8
        assert perception.values_dim == 32

        temporal = TemporalConfig()
        assert temporal.seq_len == 20
        assert temporal.world_state_dim == 80
        assert temporal.observational_dropout == 0.6

        training = TrainingConfig()
        assert training.batch_size == 32
        assert training.learning_rate == 3e-4
        assert training.error_threshold == 0.35
        assert training.kl_beta == 1.0

    def test_json_round_trip(self, tmp_path: Path):
        cfg = ExperimentConfig(
            experiment_name="test_exp",
            run_id="test_001",
            seed=99,
            data=DataConfig(sequence_len=10, image_size=32),
            training=TrainingConfig(batch_size=16, max_epochs=5),
        )
        path = tmp_path / "cfg.json"
        cfg.save(str(path))

        loaded = ExperimentConfig.load(str(path))
        assert loaded.experiment_name == "test_exp"
        assert loaded.run_id == "test_001"
        assert loaded.seed == 99
        assert loaded.data.sequence_len == 10
        assert loaded.data.image_size == 32
        assert loaded.training.batch_size == 16
        assert loaded.training.max_epochs == 5

    def test_deterministic_serialization(self, tmp_path: Path):
        cfg_a = ExperimentConfig(
            experiment_name="det",
            run_id="a",
            seed=1,
            perception=PerceptionConfig(k=4, values_dim=16),
            temporal=TemporalConfig(seq_len=10),
        )
        cfg_b = ExperimentConfig(
            experiment_name="det",
            run_id="a",
            seed=1,
            perception=PerceptionConfig(k=4, values_dim=16),
            temporal=TemporalConfig(seq_len=10),
        )
        assert cfg_a.to_json() == cfg_b.to_json()

    def test_from_dict_round_trip(self):
        cfg = ExperimentConfig(
            experiment_name="dict_test",
            seed=7,
            data=DataConfig(sequence_len=8),
        )
        d = cfg.to_dict()
        restored = ExperimentConfig.from_dict(d)
        assert restored.experiment_name == "dict_test"
        assert restored.seed == 7
        assert restored.data.sequence_len == 8

    def test_individual_config_round_trip(self):
        dc = DataConfig(sequence_len=12, image_size=48)
        j = dc.to_json()
        restored = DataConfig.from_dict(json.loads(j))
        assert restored.sequence_len == 12
        assert restored.image_size == 48


# ===================================================================
# Run directory tests
# ===================================================================

class TestRunDirectory:
    def test_create_run_directory_structure(self, tmp_path: Path):
        cfg = ExperimentConfig(experiment_name="unit_test", seed=42)
        run_dir = create_run_directory(
            "unit_test", cfg, run_id="test001", runs_root=tmp_path,
        )
        assert run_dir.exists()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "environment.json").exists()
        # git metadata may be absent in test env
        assert (run_dir / "metrics").is_dir()
        assert (run_dir / "checkpoints").is_dir()
        assert (run_dir / "probes").is_dir()

    def test_config_persisted_correctly(self, tmp_path: Path):
        cfg = ExperimentConfig(
            experiment_name="persist_test",
            seed=123,
            training=TrainingConfig(batch_size=8, max_epochs=3),
        )
        create_run_directory("persist_test", cfg, run_id="p1", runs_root=tmp_path)

        loaded = load_run_config("persist_test", "p1", runs_root=tmp_path)
        assert loaded is not None
        assert loaded.seed == 123
        assert loaded.training.batch_size == 8
        assert loaded.training.max_epochs == 3

    def test_generated_run_id_is_persisted_in_config(self, tmp_path: Path):
        cfg = ExperimentConfig(experiment_name="persist_generated", seed=123)
        run_dir = create_run_directory("persist_generated", cfg, runs_root=tmp_path)

        loaded = ExperimentConfig.load(str(run_dir / "config.json"))
        assert loaded.run_id == run_dir.name
        assert loaded.experiment_name == "persist_generated"

    def test_find_run_dir_missing(self, tmp_path: Path):
        assert find_run_dir("nonexistent", "nope", runs_root=tmp_path) is None

    def test_load_run_config_missing(self, tmp_path: Path):
        assert load_run_config("missing", "x", runs_root=tmp_path) is None


# ===================================================================
# Dataset manifest tests
# ===================================================================

class TestDatasetManifest:
    def _create_dummy_rollouts(self, root: Path, count: int = 5):
        for i in range(count):
            p = root / f"rollout_{i}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            rng = np.random.RandomState(i)
            obs = rng.randint(0, 256, size=(20, 64, 64, 3), dtype=np.uint8)
            action = rng.uniform(-1, 1, size=(20, 3)).astype(np.float32)
            reward = rng.randn(20).astype(np.float32)
            done = np.zeros(20, dtype=bool)
            np.savez_compressed(p, obs=obs, action=action, reward=reward, done=done)

    def test_build_manifest_creates_valid_structure(self, tmp_path: Path):
        self._create_dummy_rollouts(tmp_path, count=5)
        manifest = build_dataset_manifest(
            tmp_path, sequence_len=10, val_ratio=0.2, shuffle_seed=42,
        )
        assert manifest["schema_version"] == 1
        assert manifest["num_files"] == 5
        assert manifest["sequence_len"] == 10
        assert "split" in manifest
        assert manifest["split"]["num_train"] + manifest["split"]["num_val"] == 5
        assert len(manifest["files"]) == 5
        for entry in manifest["files"]:
            assert "path" in entry
            assert "sha256" in entry  # default store_hashes=True
            assert "size_bytes" in entry

    def test_manifest_split_no_overlap(self, tmp_path: Path):
        self._create_dummy_rollouts(tmp_path, count=5)
        manifest = build_dataset_manifest(tmp_path, val_ratio=0.2, shuffle_seed=42)
        train_set = set(manifest["split"]["train_files"])
        val_set = set(manifest["split"]["val_files"])
        assert train_set.isdisjoint(val_set)

    def test_manifest_persistence(self, tmp_path: Path):
        self._create_dummy_rollouts(tmp_path, count=5)
        manifest = build_dataset_manifest(tmp_path, shuffle_seed=7)
        manifest_path = tmp_path / "manifest.json"
        save_manifest(manifest, manifest_path)
        assert manifest_path.exists()

        loaded = load_manifest(manifest_path)
        assert loaded["schema_version"] == 1
        assert loaded["num_files"] == 5
        assert loaded["split"]["shuffle_seed"] == 7

    def test_validate_manifest_ok(self, tmp_path: Path):
        self._create_dummy_rollouts(tmp_path, count=5)
        manifest = build_dataset_manifest(tmp_path)
        issues = validate_manifest(manifest, data_root=tmp_path)
        assert issues == [], f"Unexpected issues: {issues}"

    def test_validate_manifest_missing_file(self, tmp_path: Path):
        self._create_dummy_rollouts(tmp_path, count=3)
        manifest = build_dataset_manifest(tmp_path)
        # Add a non-existent file
        manifest["files"].append({"path": "ghost.npz", "size_bytes": 0})
        manifest["num_files"] = 4
        issues = validate_manifest(manifest, data_root=tmp_path)
        assert any("Missing" in i for i in issues)

    def test_validate_manifest_overlap(self, tmp_path: Path):
        self._create_dummy_rollouts(tmp_path, count=3)
        manifest = build_dataset_manifest(tmp_path, val_ratio=0.3)
        # Force overlap
        split = manifest["split"]
        split["train_files"] = ["rollout_0.npz", "rollout_1.npz"]
        split["val_files"] = ["rollout_1.npz", "rollout_2.npz"]
        issues = validate_manifest(manifest, data_root=tmp_path)
        assert any("overlap" in i for i in issues)

    def test_validate_manifest_hash_mismatch(self, tmp_path: Path):
        self._create_dummy_rollouts(tmp_path, count=3)
        manifest = build_dataset_manifest(tmp_path)
        (tmp_path / "rollout_0.npz").write_bytes(b"changed")

        issues = validate_manifest(manifest, data_root=tmp_path)
        assert any("Hash mismatch" in issue for issue in issues)


# ===================================================================
# Checkpoint tests
# ===================================================================

class TestCheckpointing:
    def test_structured_checkpoint_save_and_load(self, tmp_path: Path):
        cfg = ExperimentConfig(experiment_name="ckpt_test", seed=7)
        model_state = {"weight": torch.tensor([1.0, 2.0])}
        optimizer_state = {"param_groups": [], "state": {}}

        ckpt_path = save_checkpoint(
            tmp_path / "model",
            model_state=model_state,
            optimizer_state=optimizer_state,
            config=cfg,
            global_step=100,
            epoch=5,
            metrics={"loss": 0.123},
            dataset_manifest_ref="manifest_v1.json",
        )
        assert ckpt_path.exists()

        loaded = load_checkpoint(ckpt_path)
        assert loaded["schema_version"] == _CHECKPOINT_SCHEMA_VERSION
        assert not loaded["legacy"]
        assert loaded["global_step"] == 100
        assert loaded["epoch"] == 5
        assert loaded["metrics"]["loss"] == 0.123
        assert loaded["dataset_manifest_ref"] == "manifest_v1.json"
        assert loaded["config"] is not None
        assert loaded["config"].experiment_name == "ckpt_test"
        assert loaded["config"].seed == 7

        # Model state round-trips correctly
        torch.testing.assert_close(
            loaded["model_state"]["weight"],
            torch.tensor([1.0, 2.0]),
        )

    def test_legacy_bare_state_dict_load(self, tmp_path: Path):
        """Loading a bare state_dict (no schema_version) should work
        with a warning."""
        state_dict = {"weight": torch.tensor([3.0, 4.0])}
        path = tmp_path / "legacy.pt"
        torch.save(state_dict, path)

        with pytest.warns(UserWarning, match="legacy"):
            loaded = load_checkpoint(path)

        assert loaded["legacy"]
        assert loaded["schema_version"] == 1
        assert loaded["config"] is None
        torch.testing.assert_close(
            loaded["model_state"]["weight"],
            torch.tensor([3.0, 4.0]),
        )

    def test_legacy_full_state_dict_load(self, tmp_path: Path):
        """A dict with other keys but no schema_version is still legacy."""
        data = {
            "model_state": {"w": torch.tensor(1.0)},
            "optimizer_state": {"dummy": True},
        }
        path = tmp_path / "legacy_full.pt"
        torch.save(data, path)

        with pytest.warns(UserWarning, match="legacy"):
            loaded = load_checkpoint(path)

        assert loaded["legacy"]
        assert loaded["schema_version"] == 1

    def test_map_location_cpu(self, tmp_path: Path):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for map_location test")
        cfg = ExperimentConfig(experiment_name="map_test")
        model_state = {"w": torch.tensor([1.0], device="cuda:0")}
        save_checkpoint(tmp_path / "cuda_ckpt", model_state=model_state, config=cfg)

        loaded = load_checkpoint(tmp_path / "cuda_ckpt", map_location="cpu")
        assert loaded["model_state"]["w"].device.type == "cpu"


# ===================================================================
# Seeding tests
# ===================================================================

class TestSeeding:
    def test_set_seed_reproducible_python(self):
        set_seed(42)
        a = [random.random() for _ in range(5)]
        set_seed(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_set_seed_reproducible_numpy(self):
        set_seed(99)
        a = np.random.randn(5)
        set_seed(99)
        b = np.random.randn(5)
        np.testing.assert_array_equal(a, b)

    def test_set_seed_reproducible_torch(self):
        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        b = torch.randn(5)
        torch.testing.assert_close(a, b)

    def test_get_current_seed(self):
        set_seed(777)
        assert get_current_seed() == 777

    def test_get_deterministic_flag_default(self):
        set_seed(1)
        assert not get_deterministic_flag()

    def test_get_deterministic_flag_enabled(self):
        set_seed(1, deterministic=True)
        assert get_deterministic_flag()

    def test_different_seeds_differ(self):
        set_seed(1)
        a = torch.randn(5)
        set_seed(2)
        b = torch.randn(5)
        assert not torch.allclose(a, b)

    def test_seed_context_restores_state(self):
        set_seed(100)
        before = torch.randn(3)  # consume 3 values from seed 100

        with SeedContext(42):
            inside = torch.randn(3)  # deterministic inside context

        # After exit, state is restored to position after 'before' (position 4)
        after = torch.randn(3)

        # Verify: reseed, skip 3 values, then the 4th-6th match 'after'
        set_seed(100)
        _ = torch.randn(3)  # positions 1-3
        expected_after = torch.randn(3)  # positions 4-6
        torch.testing.assert_close(after, expected_after)

        # Inside the context, state was deterministic
        with SeedContext(42):
            inside_again = torch.randn(3)
        torch.testing.assert_close(inside, inside_again)

    def test_seed_context_restores_seed_metadata(self):
        set_seed(100, deterministic=False)
        with SeedContext(42, deterministic=True):
            assert get_current_seed() == 42
            assert get_deterministic_flag()

        assert get_current_seed() == 100
        assert not get_deterministic_flag()
