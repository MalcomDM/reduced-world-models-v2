# Experiment Artifacts — Stage 0.5 Infrastructure

## Purpose

This document describes the structured experiment artifacts introduced in
Stage 0.5: typed configuration, run directories, dataset manifests,
structured checkpoints, and seeding.  These artifacts make every future
experiment reproducible, identifiable, resumable, and comparable.

This infrastructure is additive — it does not change the model architecture,
transition semantics, or training loop internals.

---

## 1. Run Directory Layout

```
runs/<experiment_name>/<run_id>/
    config.json              # Resolved ExperimentConfig (JSON)
    environment.json         # Python, PyTorch, CUDA, platform info
    git_metadata.json        # Commit hash + dirty flag (if available)
    metrics/                 # Per-epoch or per-step metric logs
    checkpoints/             # Structured checkpoint .pt files
    probes/                  # Fixed probe-set outputs
```

Run creation:

```python
from rwm.config.experiment_config import ExperimentConfig
from rwm.utils.run_directory import create_run_directory

cfg = ExperimentConfig(experiment_name="my_exp", seed=42)
run_dir = create_run_directory("my_exp", cfg, run_id="exp001")
```

Without an explicit ``run_id``, a timestamp-based unique ID is generated.

---

## 2. Configuration System

Configuration is organized into six frozen dataclass groups, each
serializable to/from JSON without external dependencies:

| Group | Fields | Defaults matching |
|-------|--------|-------------------|
| ``DataConfig`` | dataset_dir, sequence_len, image_size, val_ratio, include_done_in_train, num_workers, pin_memory | ``config.py`` |
| ``PerceptionConfig`` | conv_filters/kernel/strides/paddings, patch_size/stride, token_dim, k, query_dim, values_dim | ``config.py`` |
| ``TemporalConfig`` | seq_len, world_state_dim, observational_dropout, warmup_steps, ffn_mult, transformer_dropout | ``config.py`` |
| ``ControllerConfig`` | action_dim, hidden_dim, noise_std, rollout_len, n_rollouts, top_k, positive_threshold, memory_batch, memory_size | ``config.py`` |
| ``TrainingConfig`` | batch_size, learning_rate, weight_decay, max_epochs, alpha, beta, warmup_steps, error_threshold, kl_beta | ``config.py`` |
| ``ExperimentConfig`` | experiment_name, run_id, seed, + all above sub-configs | — |

### JSON round-trip

```python
cfg = ExperimentConfig(experiment_name="test", seed=7)
cfg.save("config.json")          # deterministic JSON with sorted keys
loaded = ExperimentConfig.load("config.json")
assert loaded == cfg_from_dict   # structural equality
```

The ``--seed`` CLI flag on ``rwm train-rwm`` creates a run directory with
persisted ``config.json`` before training begins.

---

## 3. Dataset Manifest

A dataset manifest records the exact rollout files and their train/val
partition used by a run.

### Schema (version 1)

```json
{
    "schema_version": 1,
    "data_root": "/absolute/path/to/data",
    "created_at": "2026-07-12T12:00:00Z",
    "num_files": 50,
    "sequence_len": 16,
    "image_size": 64,
    "include_done": false,
    "split": {
        "method": "episode_safe",
        "val_ratio": 0.2,
        "shuffle_seed": 42,
        "num_train": 40,
        "num_val": 10,
        "train_files": ["rel/path/rollout_0.npz", ...],
        "val_files":   ["rel/path/rollout_42.npz", ...]
    },
    "files": [
        {"path": "rel/path/rollout_0.npz", "size_bytes": 12345, "sha256": "abc..."},
        ...
    ]
}
```

### Usage

```python
from rwm.utils.dataset_manifest import build_dataset_manifest, save_manifest, validate_manifest

manifest = build_dataset_manifest(
    data_root=Path("data/rollouts/rwm_deterministic/scenario_0"),
    sequence_len=16,
    val_ratio=0.2,
    shuffle_seed=42,
)
issues = validate_manifest(manifest)  # empty = valid
save_manifest(manifest, Path("manifest.json"))
```

### Validation checks

- Schema version matches the expected value.
- Every referenced file exists on disk.
- Train and val file sets are disjoint.
- ``train_files + val_files == num_files``.

---

## 4. Structured Checkpoints

### Format (version 2)

```python
{
    "schema_version": 2,
    "model_state":          { ... state_dict ... },
    "optimizer_state":      { ... } or None,
    "scheduler_state":      { ... } or None,
    "global_step":          1234,
    "epoch":                42,
    "config":               { ... ExperimentConfig.to_dict() ... },
    "metrics":              {"loss": 0.123, "mae_cum": 0.5},
    "rng_state":            {"python": ..., "numpy": ..., "torch": ...},
    "dataset_manifest_ref": "manifest_v1.json",
}
```

### Save and load

```python
from rwm.utils.checkpointing import save_checkpoint, load_checkpoint

ckpt_path = save_checkpoint(
    path=run_dir / "checkpoints" / "model",
    model_state=model.state_dict(),
    config=cfg,
    optimizer_state=optimizer.state_dict(),
    epoch=5,
    metrics={"loss": 0.123},
)

loaded = load_checkpoint(ckpt_path)
model.load_state_dict(loaded["model_state"])
print(loaded["config"].experiment_name)   # restored ExperimentConfig
```

### Legacy compatibility

Loading a bare ``state_dict`` (version 1 or no schema) produces a warning
and returns ``{"legacy": True, "model_state": ..., "config": None, ...}``.
This ensures existing checkpoints remain readable.

```python
loaded = load_checkpoint(Path("legacy.pt"))
assert loaded["legacy"]
assert loaded["schema_version"] == 1
```

Use ``map_location="cpu"`` to load a GPU checkpoint on a CPU-only machine.

---

## 5. Seeding

### Quick start

```python
from rwm.utils.seeding import set_seed

set_seed(42)                     # seeds Python, NumPy, PyTorch
set_seed(42, deterministic=True) # prefer deterministic cuDNN behavior
```

### Context manager

```python
from rwm.utils.seeding import SeedContext

with SeedContext(99):
    x = torch.randn(3)  # deterministic
```

The context saves and restores all three RNG states on entry/exit.

### Tracking

```python
from rwm.utils.seeding import get_current_seed, get_deterministic_flag

set_seed(777)
assert get_current_seed() == 777
assert not get_deterministic_flag()
```

The current seed is recorded in ``config.json`` under the ``seed`` field
of ``ExperimentConfig``.

The deterministic flag configures cuDNN deterministic/benchmark behavior; it
does not guarantee that every CUDA operation used by PyTorch is deterministic.

---

## 6. Current CLI Integration

The ``rwm train-rwm`` command accepts two new optional flags:

- ``--seed``: set random seed and create a structured run directory.
- ``--run-id``: explicit run identifier (used with ``--seed``).

Without ``--seed``, the command behaves exactly as before (backward
compatible).

### Structured Checkpoint Integration (Stage 1)

The ``rwm train-rwm`` command now produces structured checkpoints when
``--seed`` is provided:

- ``checkpoint_best.pt`` — structured checkpoint with model state, optimizer
  state, config, metrics, RNG state, and dataset manifest reference.
- ``checkpoint_latest.pt`` — same format, updated every epoch.
- ``best_world_model.pt`` — legacy bare ``state_dict`` for backward
  compatibility.

A dataset manifest (``dataset_manifest.json``) is built and saved in the
run directory before training begins.

Without ``--seed``, the command behaves identically to previous versions
(legacy output directory, bare ``state_dict`` only).
