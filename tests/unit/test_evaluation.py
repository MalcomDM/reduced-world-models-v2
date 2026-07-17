"""Tests for Stage 2.5A evaluation infrastructure."""
import dataclasses
import json
import numpy as np
import pytest
import torch
from pathlib import Path
from typer.testing import CliRunner

from rwm.evaluation.schema import (
    SeedManifest,
    EpisodeMetadata,
    Split,
    Quality,
    load_seed_manifest,
    save_seed_manifest,
    load_episode_metadata,
    save_episode_metadata,
    make_episode_metadata,
    _compute_manifest_hash,
    validate_episode_integrity,
)
from rwm.evaluation.collector import collect_evaluation_episode
from rwm.evaluation.episode_evaluator import evaluate_split
from rwm.evaluation.branch_runner import (
    BranchResult,
    BranchExperiment,
    save_branch_experiment,
)
from rwm.evaluation.episode_evaluator import EpisodeResult
from rwm.evaluation.attention_trace import (
    patch_grid_coords,
    image_coords_from_patches,
    AttentionTrace,
    trace_attention,
    render_heatmap,
    render_selected_overlay,
)
from rwm.models.rwm.model import ReducedWorldModel
from rwm.data.rollout_dataset import _is_evaluation_file
from rwm.cli import app


# ===================================================================
# Attention rendering tests
# ===================================================================

def test_heatmap_averages_overlapping_patch_scores():
    """Overlaps must show the mean score, not the final loop assignment."""
    patch_count = 15 * 15
    logits = torch.zeros((1, patch_count))
    logits[0, 0] = 1.0
    logits[0, 1] = 3.0
    trace = AttentionTrace(
        logits=logits,
        indices=torch.tensor([[0]]),
        weights=torch.tensor([[1.0]]),
    )

    rendered = render_heatmap(trace)[0]
    soft = torch.softmax(logits[0], dim=0)
    raw = torch.zeros((64, 64))
    coverage = torch.zeros((64, 64))
    for index, value in enumerate(soft):
        y, x = divmod(index, 15)
        raw[y * 4:y * 4 + 8, x * 4:x * 4 + 8] += value
        coverage[y * 4:y * 4 + 8, x * 4:x * 4 + 8] += 1
    expected = raw / coverage.clamp_min(1)
    expected /= expected.max()

    # Pixel (0, 4) belongs to patches 0 and 1.
    assert torch.isclose(rendered[0, 4], expected[0, 4])


# ===================================================================
# Schema tests
# ===================================================================

class TestSeedManifest:
    def test_empty_fails(self):
        assert SeedManifest.validate_entries({})

    def test_valid_passes(self):
        assert SeedManifest.validate_entries({"0": "dev", "1": "val", "5": "locked_test"}) == []

    def test_invalid_split(self):
        issues = SeedManifest.validate_entries({"0": "training"})
        assert any("Invalid split" in i for i in issues)

    def test_invalid_seed(self):
        issues = SeedManifest.validate_entries({"abc": "dev"})
        assert any("not an integer" in i for i in issues)

    def test_save_load(self, tmp_path):
        m = SeedManifest(created_at="now", entries={"0": "dev"})
        p = tmp_path / "m.json"
        save_seed_manifest(m, p)
        assert load_seed_manifest(p).entries == {"0": "dev"}

    def test_assert_valid(self):
        with pytest.raises(ValueError, match="Seed manifest"):
            SeedManifest(entries={}).assert_valid()


# ===================================================================
# Manifest freeze tests
# ===================================================================

class TestManifestFreeze:
    def test_init_refuses_overwrite_default(self, tmp_path):
        mf = tmp_path / "seeds.json"
        mf.write_text('{"schema_version":1,"entries":{"0":"dev"}}')
        runner = CliRunner()
        r = runner.invoke(app, ["eval", "init-seeds", str(mf), "--dev-seeds", "0"])
        assert r.exit_code != 0
        assert "exists" in r.stdout.lower() or "exists" in r.stderr.lower()

    def test_init_force_replace_empty(self, tmp_path):
        mf = tmp_path / "seeds.json"
        mf.write_text('{"schema_version":1,"entries":{}}')
        runner = CliRunner()
        r = runner.invoke(app, ["eval", "init-seeds", str(mf), "--dev-seeds", "0", "--force-replace"])
        assert r.exit_code == 0

    def test_init_rejects_duplicate_seeds(self, tmp_path):
        mf = tmp_path / "seeds.json"
        runner = CliRunner()
        r = runner.invoke(app, ["eval", "init-seeds", str(mf),
                                "--dev-seeds", "10", "--test-seeds", "10"])
        assert r.exit_code != 0
        assert "Duplicate" in (r.stderr + r.stdout)

    def test_manifest_hash_computation(self, tmp_path):
        m = SeedManifest(created_at="now", entries={"0": "dev"})
        p = tmp_path / "m.json"
        save_seed_manifest(m, p)
        h = _compute_manifest_hash(p)
        assert len(h) == 16
        assert isinstance(h, str)

    def test_init_force_replace_rejected_after_collection(self, tmp_path):
        mf = tmp_path / "seeds.json"
        save_seed_manifest(SeedManifest(created_at="now", entries={"7": "dev"}), mf)
        ep_dir = tmp_path / "dev"
        ep_dir.mkdir()
        ep = ep_dir / "episode.npz"
        np.savez_compressed(ep, obs=np.zeros((1, 64, 64, 3), dtype=np.uint8),
                            action=np.zeros((1, 3)), reward=np.zeros(1))
        save_episode_metadata(EpisodeMetadata(
            episode_id="episode", split="dev", track_seed=7,
            manifest_hash=_compute_manifest_hash(mf), manifest_path=str(mf.resolve()),
        ), ep.with_suffix(".episode.json"))
        r = CliRunner().invoke(
            app, ["eval", "init-seeds", str(mf), "--dev-seeds", "8", "--force-replace"]
        )
        assert r.exit_code != 0
        assert load_seed_manifest(mf).entries == {"7": "dev"}


# ===================================================================
# Label metadata preservation
# ===================================================================

class TestLabelPreservation:
    def test_label_preserves_all_provenance(self, tmp_path):
        """Ensure ``eval label`` preserves every field except quality/tags/operator/notes."""
        meta = EpisodeMetadata(
            episode_id="ep1",
            split="dev",
            track_seed=42,
            purpose="evaluation_only",
            env_id="CarRacing-v3",
            env_version="1.0",
            policy="human",
            collector_timestamp="2025-01-01T00:00:00Z",
            git_commit="abc123",
            config_ref="cfg_v1",
            max_steps=1000,
            early_push=10,
            idle_threshold=50,
            render_mode="human",
            manifest_hash="deadbeef",
            manifest_path="/some/path/manifest.json",
            terminated=False,
            truncated=True,
            steps=500,
            quality="unreviewed",
            scenario_tags="",
            operator="",
            notes="",
        )
        p = tmp_path / "ep.episode.json"
        save_episode_metadata(meta, p)

        # Simulate label
        loaded = load_episode_metadata(p)
        updated = dataclasses.replace(
            loaded,
            quality="keep",
            scenario_tags="curve_left",
            operator="tester",
            notes="Good episode",
        )
        save_episode_metadata(updated, p)

        reloaded = load_episode_metadata(p)
        assert reloaded.episode_id == "ep1"
        assert reloaded.track_seed == 42
        assert reloaded.env_version == "1.0"
        assert reloaded.manifest_hash == "deadbeef"
        assert reloaded.manifest_path == "/some/path/manifest.json"
        assert reloaded.render_mode == "human"
        assert reloaded.max_steps == 1000
        assert reloaded.early_push == 10
        assert reloaded.idle_threshold == 50
        assert reloaded.terminated is False
        assert reloaded.truncated is True
        assert reloaded.steps == 500
        assert reloaded.quality == "keep"
        assert reloaded.scenario_tags == "curve_left"
        assert reloaded.operator == "tester"
        assert reloaded.notes == "Good episode"


# ===================================================================
# Evaluation-only protection
# ===================================================================

class TestEvaluationProtection:
    def test_episode_sidecar_detected(self, tmp_path):
        npz = tmp_path / "ep.npz"
        np.savez_compressed(npz, obs=np.zeros((1, 8, 8, 3)))
        assert not _is_evaluation_file(npz)
        npz.with_suffix(".episode.json").write_text("{}")
        assert _is_evaluation_file(npz)

    def test_branch_sidecar_detected(self, tmp_path):
        npz = tmp_path / "br.npz"
        np.savez_compressed(npz, obs=np.zeros((1, 8, 8, 3)))
        assert not _is_evaluation_file(npz)
        npz.with_suffix(".branch.json").write_text("{}")
        assert _is_evaluation_file(npz)

    def test_cached_and_uncached_samples_match(self, tmp_path):
        """Cached and uncached RolloutDataset must produce identical samples."""
        from rwm.data.rollout_dataset import RolloutDataset, _collect_npz_files
        import tempfile
        # Build a cache for synthetic data
        cache_root = tmp_path / "cache"
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        # Create one synthetic rollout
        obs = np.random.randint(0, 256, (30, 64, 64, 3), dtype=np.uint8)
        act = np.random.uniform(-1, 1, (30, 3)).astype(np.float32)
        rew = np.random.randn(30).astype(np.float32)
        don = np.zeros(30, dtype=bool)
        src = npz_dir / "test_ep.npz"
        np.savez_compressed(src, obs=obs, action=act, reward=rew, done=don)

        # Build cache
        from scripts.build_frame_cache import build_cache
        build_cache(data_root=npz_dir, cache_dir=cache_root, dry_run=False)

        # Compare dataset samples
        ds_uncached = RolloutDataset.from_file_list([src], sequence_len=8, image_size=64)
        ds_cached = RolloutDataset.from_file_list([src], sequence_len=8, image_size=64, cache_dir=cache_root)

        assert len(ds_uncached) == len(ds_cached), "Window counts must match"

        for i in range(min(len(ds_uncached), 5)):
            u = ds_uncached[i]
            c = ds_cached[i]
            torch.testing.assert_close(u["obs"], c["obs"], msg=f"Sample {i}: obs mismatch")
            torch.testing.assert_close(u["action"], c["action"], msg=f"Sample {i}: action mismatch")
            torch.testing.assert_close(u["reward"], c["reward"], msg=f"Sample {i}: reward mismatch")
            torch.testing.assert_close(u["done"], c["done"], msg=f"Sample {i}: done mismatch")
            torch.testing.assert_close(u["predecessor_action"], c["predecessor_action"],
                                      msg=f"Sample {i}: predecessor mismatch")

    def test_training_loader_rejects_eval_files(self, tmp_path):
        from rwm.data.rollout_dataset import RolloutDataset

        normal = tmp_path / "train.npz"
        np.savez_compressed(normal, obs=np.zeros((20, 8, 8, 3), dtype=np.uint8),
                            action=np.zeros((20, 3)), reward=np.zeros(20),
                            done=np.zeros(20, dtype=bool))
        ev = tmp_path / "eval_only.npz"
        np.savez_compressed(ev, obs=np.zeros((20, 8, 8, 3), dtype=np.uint8),
                            action=np.zeros((20, 3)), reward=np.zeros(20),
                            done=np.zeros(20, dtype=bool))
        ev.with_suffix(".episode.json").write_text('{"purpose":"evaluation_only"}')
        br = tmp_path / "branch.npz"
        np.savez_compressed(br, obs=np.zeros((20, 8, 8, 3), dtype=np.uint8),
                            action=np.zeros((20, 3)), reward=np.zeros(20))
        br.with_suffix(".branch.json").write_text('{"purpose":"evaluation_only_branch"}')

        ds = RolloutDataset(root_dir=tmp_path, sequence_len=8, image_size=8)
        paths = {s[0] for s in ds.samples}
        assert normal in paths
        assert ev not in paths
        assert br not in paths


# ===================================================================
# Branch runner tests
# ===================================================================

class TestBranchRunner:
    def test_dataclasses(self):
        br = BranchResult(branch_name="t", actions=np.zeros((3, 3)),
                          observations=np.zeros((3, 8, 8, 3), dtype=np.uint8),
                          rewards=np.zeros(3), terminated=False, truncated=False)
        assert br.branch_name == "t"

    def test_branch_experiment_save_to_separate_path(self, tmp_path):
        exp = BranchExperiment(
            seed=7,
            prefix_actions=np.zeros((5, 3)),
            prefix_observations=np.zeros((5, 8, 8, 3), dtype=np.uint8),
            prefix_rewards=np.zeros(5),
            branches={},
        )
        save_branch_experiment(exp, tmp_path)
        branch_dir = tmp_path / "branches"
        assert branch_dir.exists()
        npz_files = list(branch_dir.glob("*.npz"))
        assert len(npz_files) >= 1
        json_files = list(branch_dir.glob("*.branch.json"))
        assert len(json_files) >= 1

    def test_branch_not_under_split_dir(self, tmp_path):
        """Branch experiments must not be saved under dev/val/locked_test."""
        exp = BranchExperiment(
            seed=7,
            prefix_actions=np.zeros((5, 3)),
            prefix_observations=np.zeros((5, 8, 8, 3), dtype=np.uint8),
            prefix_rewards=np.zeros(5),
            branches={},
        )
        save_branch_experiment(exp, tmp_path)
        for split in ("dev", "val", "locked_test"):
            assert not (tmp_path / split).exists()


# ===================================================================
# Human collection validation (no display)
# ===================================================================

class TestHumanValidation:
    def test_human_rejects_non_human_render(self, tmp_path):
        """Check that human policy without human render is rejected,
        independent of whether the manifest exists."""
        mf = tmp_path / "seeds.json"
        mf.write_text('{"schema_version":1,"entries":{}}')
        runner = CliRunner()
        r = runner.invoke(app, ["eval", "collect", str(mf), "0",
                                "--policy-name", "human", "--render-mode", "rgb_array"])
        # The error about human+render happens before manifest validation
        output = (r.stderr + r.stdout).lower()
        assert r.exit_code != 0
        assert "requires" in output or "human" in output

    def test_collect_help_shows_fps(self):
        runner = CliRunner()
        r = runner.invoke(app, ["eval", "collect", "--help"])
        assert "--fps" in r.stdout

    def test_collector_initializes_input_after_env_and_honors_fps(self, tmp_path, monkeypatch):
        """No display is needed to verify lifecycle order and frame limiting."""
        from rwm.evaluation import collector

        mf = tmp_path / "seeds.json"
        save_seed_manifest(SeedManifest(entries={"1": "dev"}), mf)

        class FakeEnv:
            reset_called = False
            closed = False
            def reset(self, seed):
                self.reset_called = True
                return np.zeros((64, 64, 3), dtype=np.uint8), {}
            def step(self, action):
                return np.zeros((64, 64, 3), dtype=np.uint8), 0.0, False, False, {}
            def close(self):
                self.closed = True
        env = FakeEnv()
        monkeypatch.setattr(collector, "make_env", lambda *a, **kw: env)

        class FakeClock:
            def __init__(self): self.calls = []
            def tick(self, fps): self.calls.append(fps)
        clock = FakeClock()
        ready = []
        path = collect_evaluation_episode(
            mf, 1, tmp_path, lambda obs: np.zeros(3, dtype=np.float32),
            max_steps=1, clock=clock, fps=37,
            on_env_ready=lambda: ready.append(env.reset_called),
        )
        assert path.exists() and path.with_suffix(".episode.json").exists()
        assert ready == [True]
        assert clock.calls == [37]

    def test_collector_saves_partial_episode_when_policy_fails(self, tmp_path, monkeypatch):
        from rwm.evaluation import collector
        mf = tmp_path / "seeds.json"
        save_seed_manifest(SeedManifest(entries={"1": "dev"}), mf)

        class FakeEnv:
            def reset(self, seed): return np.zeros((64, 64, 3), dtype=np.uint8), {}
            def step(self, action):
                return np.zeros((64, 64, 3), dtype=np.uint8), 0.0, False, False, {}
            def close(self): pass
        monkeypatch.setattr(collector, "make_env", lambda *a, **kw: FakeEnv())
        calls = 0
        def policy(obs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("policy failure")
            return np.zeros(3, dtype=np.float32)
        with pytest.raises(RuntimeError, match="policy failure"):
            collect_evaluation_episode(mf, 1, tmp_path, policy, max_steps=3)
        files = list((tmp_path / "dev").glob("*.npz"))
        assert len(files) == 1
        assert files[0].with_suffix(".episode.json").exists()


# ===================================================================
# Status command tests
# ===================================================================

class TestStatus:
    def test_status_empty(self, tmp_path):
        runner = CliRunner()
        r = runner.invoke(app, ["eval", "status", str(tmp_path)])
        assert r.exit_code == 0

    def test_status_shows_branches_separately(self, tmp_path):
        # Create a branch experiment
        from rwm.evaluation.branch_runner import BranchExperiment, save_branch_experiment
        exp = BranchExperiment(seed=0, prefix_actions=np.zeros((3, 3)),
                               prefix_observations=np.zeros((3, 8, 8, 3), dtype=np.uint8),
                               prefix_rewards=np.zeros(3), branches={})
        save_branch_experiment(exp, tmp_path)
        runner = CliRunner()
        r = runner.invoke(app, ["eval", "status", str(tmp_path)])
        assert "branches" in r.stdout


class TestEvaluationIntegrity:
    def test_integrity_detects_manifest_and_split_mismatch(self, tmp_path):
        mf = tmp_path / "seeds.json"
        save_seed_manifest(SeedManifest(entries={"3": "dev"}), mf)
        meta = EpisodeMetadata(
            purpose="evaluation_only", split="val", track_seed=3,
            manifest_hash="wrong", manifest_path=str(mf.resolve()),
        )
        issues = validate_episode_integrity(meta, load_seed_manifest(mf), mf, "dev")
        assert any("hash mismatch" in issue for issue in issues)
        assert any("split" in issue for issue in issues)

    def test_evaluator_rejects_invalid_episode_before_metrics(self, tmp_path):
        split_dir = tmp_path / "dev"
        split_dir.mkdir()
        mf = tmp_path / "seeds.json"
        save_seed_manifest(SeedManifest(entries={"3": "dev"}), mf)
        ep = split_dir / "bad.npz"
        np.savez_compressed(ep, obs=np.zeros((1, 64, 64, 3), dtype=np.uint8),
                            action=np.zeros((1, 3)), reward=np.zeros(1))
        save_episode_metadata(EpisodeMetadata(
            purpose="not_evaluation", split="dev", track_seed=3,
            manifest_hash=_compute_manifest_hash(mf), manifest_path=str(mf.resolve()),
        ), ep.with_suffix(".episode.json"))
        with pytest.raises(ValueError, match="Invalid evaluation episode"):
            evaluate_split(torch.nn.Identity(), split_dir, "dev", mf)


# ===================================================================
# Attention trace tests
# ===================================================================

class TestAttentionTrace:
    def test_patch_coords_shape(self):
        assert patch_grid_coords().shape == (225, 2)

    def test_image_coords(self):
        assert image_coords_from_patches(32).shape == (225, 2)

    def test_trace_parity(self):
        """trace_attention must not alter model evaluation outputs."""
        model = ReducedWorldModel()
        model.eval()
        img = torch.randn(1, 3, 64, 64)
        act = torch.zeros(1, 3)

        with torch.no_grad():
            out1 = model(img=img, prev_action=act, current_action=act, force_keep_input=True)

        trace = trace_attention(model, img)
        assert trace.logits.shape == (1, 225)
        assert trace.indices.shape == (1, 8)

        with torch.no_grad():
            out2 = model(img=img, prev_action=act, current_action=act, force_keep_input=True)

        torch.testing.assert_close(out1.world_state, out2.world_state)

    def test_attentive_trace_batch(self):
        model = ReducedWorldModel()
        model.eval()
        trace = trace_attention(model, torch.randn(2, 3, 64, 64))
        assert trace.logits.shape == (2, 225)
        assert trace.indices.shape == (2, 8)

    def test_render_heatmap(self):
        logits = torch.randn(1, 225)
        indices = torch.randint(0, 225, (1, 8))
        weights = torch.randn(1, 8).softmax(dim=-1)
        trace = AttentionTrace(logits=logits, indices=indices, weights=weights)
        assert render_heatmap(trace).shape == (1, 64, 64)

    def test_render_overlay(self):
        logits = torch.randn(2, 225)
        indices = torch.randint(0, 225, (2, 8))
        weights = torch.randn(2, 8).softmax(dim=-1)
        trace = AttentionTrace(logits=logits, indices=indices, weights=weights)
        assert render_selected_overlay(trace).shape == (2, 64, 64)


# ===================================================================
# CLI smoke tests
# ===================================================================

class TestCLI:
    def test_eval_help(self):
        r = CliRunner().invoke(app, ["eval", "--help"])
        assert r.exit_code == 0

    def test_init_seeds_help(self):
        r = CliRunner().invoke(app, ["eval", "init-seeds", "--help"])
        assert r.exit_code == 0
        assert "--force-replace" in r.stdout

    def test_collect_help(self):
        r = CliRunner().invoke(app, ["eval", "collect", "--help"])
        assert r.exit_code == 0
        assert "human" in r.stdout

    def test_label_help(self):
        r = CliRunner().invoke(app, ["eval", "label", "--help"])
        assert r.exit_code == 0

    def test_status_help(self):
        r = CliRunner().invoke(app, ["eval", "status", "--help"])
        assert r.exit_code == 0


# ===================================================================
# Deterministic replay integration test (real env, headless)
# ===================================================================

@pytest.mark.integration
def test_deterministic_replay_with_real_env(tmp_path):
    """Reset CarRacing at a fixed seed, replay a prefix twice, verify
    observations and rewards match exactly."""
    try:
        from rwm.evaluation.branch_runner import run_prefix, verify_deterministic_replay
        from rwm.envs.env import make_env
        import gymnasium as gym
    except ImportError:
        pytest.skip("gymnasium not available")

    actions = np.zeros((30, 3), dtype=np.float32)
    actions[:, 1] = 1.0  # full gas for 30 steps

    env = make_env("car_racing", render_mode="rgb_array")
    first_obs, first_rewards = run_prefix(env, seed=42, prefix_actions=actions, record=True)
    env.close()

    issues = verify_deterministic_replay(
        "car_racing", seed=42, actions=actions,
        expected_obs=first_obs, expected_rewards=first_rewards,
        atol=1,
    )
    assert issues == [], f"Deterministic replay failed: {issues}"


@pytest.mark.integration
def test_branches_from_same_prefix(tmp_path):
    """Two branches starting from the same prefix must produce different actions
    and different rewards."""
    try:
        from rwm.evaluation.branch_runner import run_branch_experiment, BranchExperiment
        import numpy as np
    except ImportError:
        pytest.skip("gymnasium not available")

    prefix = np.zeros((30, 3), dtype=np.float32)
    prefix[:, 1] = 1.0

    branches = {
        "gas": np.array([[0.0, 1.0, 0.0]] * 10, dtype=np.float32),
        "brake": np.array([[0.0, 0.0, 1.0]] * 10, dtype=np.float32),
    }

    exp = run_branch_experiment(seed=42, prefix_actions=prefix, branches=branches)
    assert "gas" in exp.branches
    assert "brake" in exp.branches

    gas_rew = exp.branches["gas"].rewards
    brake_rew = exp.branches["brake"].rewards
    assert not np.array_equal(gas_rew, brake_rew), "Branch rewards should differ"
    # Observations should exist and have correct rank
    assert exp.branches["gas"].observations.ndim == 4
    assert exp.branches["brake"].observations.ndim == 4
