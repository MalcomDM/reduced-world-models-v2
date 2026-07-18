# Reduced World Models — Structural Refactor Plan

## Purpose

This document is the engineering plan for moving the current repository from
the legacy LSTM and behavior-memory controller toward the architecture defined
in `../technical_definitions.md`.

The post-Stage-2 theory/evidence audit and matched-ablation protocol are in
`architecture_validation_plan.md`. That document is the source of truth for
the Stage 2.5 gates; this file retains the larger implementation sequence.

The project is not starting from zero. The legacy LSTM path previously achieved
acceptable end-to-end reward prediction. The immediate goal is to restore that
validated checkpoint using the causal Transformer and the new shared controller
trunk before implementing Actor-Critic learning.

## Desired Structure

```text
observation o_t
    → reduced perception p_t
    + previous action a_{t-1} + causal history
    → pre-action belief b_t
    → shared controller trunk u_t
        ├── Actor head: action distribution π(a_t | b_t)
        ├── Critic head: expected future return V(b_t)
        └── reward head: R(b_t, a_t) = predicted r_{t+1}
```

The reward head is colocated with Actor and Critic so direct reward supervision
anchors their shared interpretation. All three losses may eventually shape the
temporal and perceptual blocks. That full gradient flow is enabled only after
each subsystem passes an isolated checkpoint.

## Detected Problems

### Temporal migration

- `CausalTransformer` exists, but its file name is misspelled
  `casusal_transformer.py`.
- Active controller replay and manual-evaluation paths still call the legacy
  LSTM `h_prev/c_prev` and `world_rnn.rnn_cell` interfaces.
- Legacy `WorldRNN` and `PatchRNN` tests pass but do not validate the active
  causal Transformer path.
- Temporal history construction is duplicated between the model and rollout
  simulator.
- Training evaluates the growing Transformer history once per time step instead
  of processing the full sequence in parallel.

### Credit assignment and stochastic perception

- `ReducedWorldModel._append_token()` runs under `torch.no_grad()`, so direct
  reward gradients stop before perception.
- Encoder and tokenizer are evaluated twice per training step: once for KL and
  once for reward prediction.
- The two passes can sample different stochastic tokens.
- Evaluation disables observational dropout but tokenizer/Top-K determinism
  must be tested explicitly.

### Transition semantics

- The collector stores an observation, the action taken from it, and the reward
  returned by that action at the same array index.
- Training currently predicts `reward[t]` using `observation[t]` and a previous
  action initialized to zero, which appears shifted.
- The repository lacks a synthetic test proving action/reward alignment.
- Episode termination and padding masks are not part of the temporal training
  contract.

### Controller and imagination

- The current controller is deterministic action regression, not Actor-Critic.
- The controller trainer imitates actions from only positive imagined rollouts.
- Behavior memory remains central despite being deprioritized by the design.
- Exact hashes of changing latent states are unsuitable for semantic replay.
- Imagined rollouts execute under `torch.no_grad()`, convert rewards to Python
  floats, and detach states, so they cannot train Actor or Critic end to end.
- Rollout warmup mixes NumPy and Torch actions.

### Validation and experiment control

- Current checkpoints are selected using training loss instead of held-out
  validation performance.
- Unit tests do not cover `CausalTransformer` or the complete active model.
- Constant zero-action/zero-reward tests cannot detect transition misalignment
  or missing action conditioning.
- No fixed probe set measures representation drift.
- No metric detects imagined-reward exploitation by comparing imagined and real
  policy performance.

## Critical Blocking Gaps

The following are hard gates, not backlog suggestions:

1. **Transition contract:** reward/action indexing must be proven before reward
   training results can be trusted.
2. **End-to-end reward gradients:** the causal model must propagate reward loss
   into every intended perception component.
3. **Action-conditioned imagination:** different action sequences from the same
   state must produce meaningfully different state/reward trajectories.
4. **Deterministic validation:** identical inputs and actions must produce
   identical evaluation outputs.
5. **Actor-Critic isolation:** Actor, Critic, distributions, and GAE must work
   with a frozen world model before behavior gradients are allowed upstream.
6. **Reality check:** improved imagined return must correlate with improved real
   environment return before progressive collection is enabled.

## Stage 0 — Freeze Contracts and Historical Baseline

### Completed Work

- **Transition contract** (`docs/contracts/transition_contract.md`): documented exact
  semantics of every rollout array index, training-loop action shift, dataset
  window semantics, done-flag ambiguity, and broken LSTM call sites.
- **Typed structures** (`src/rwm/types.py`): added `TemporalModelOutput`,
  `ProbeSet`, `EpisodeSamples`, `TrainValSplit` types with full shape and
  semantic docstrings.  Extended `RolloutSample` docstring with transition
  contract reference.
- **Model docstrings**: added parameter/shape documentation to
  `ReducedWorldModel.forward()` and `CausalTransformer.forward()`.
- **Synthetic regression test** (`tests/unit/test_transition_alignment.py`):
  deterministic spy tests record the actions supplied to the active trainer;
  they prove `a_prev=zeros` at `t=0` and `a_prev=action[t-1]` later, while the
  target remains `reward[t]`. No optimisation-dependent threshold is used.
- **Episode-safe split** (`src/rwm/data/rollout_dataset.py`):
  `episode_safe_train_val_split()`, `build_train_val_datasets()`, and
  `RolloutDataset.from_file_list()` classmethod.
- **Probe set** (`src/rwm/utils/probe_set.py`): deterministic observation/action
  batch generator, save/load helpers, default probe (8×64×64×3 + 8×3).
- **Legacy evidence** (recorded below): no surviving checkpoints, metrics, or
  logs found; legacy scripts under `scripts/` import obsolete `app.*` package.

### Legacy Baseline Evidence

The following were searched and confirmed **absent** from the repository:

| Artifact | Search path | Result |
|----------|-------------|--------|
| `.pt` checkpoints | `**/*.pt` | None |
| CSV logs | `**/*.csv` | None |
| JSONL stats | `**/*.jsonl` | None |
| Pickle memory dumps | `**/*.pkl` | None |
| Legacy `runs/` output | `runs/` | Empty |

Legacy LSTM reward metrics mentioned in earlier documentation
(`runs/rwm/test1/weights/best_loss_0.3305.pt`, etc.) refer to paths that do
not exist in the current worktree or committed history.  The git log contains
only structural commits; no checkpoint artifacts were ever tracked.

**Legacy scripts** (`scripts/`) import the obsolete `app.*` package and are
non-functional without migration:

- `scripts/train/train_world_model.py`
- `scripts/train/train_image_recons.py`
- `scripts/script_tests/stest_rwm_on_dataset.py`
- `scripts/script_tests/stest_rwm_on_episode.py`
- `scripts/explore/explore_rollouts.py`
- `scripts/explore/inspect_params.py`
- `scripts/explore/manual_play.py`
- `scripts/generate_data.py`

These are left untouched per the preservation rule.

**Broken call sites** (legacy LSTM interface on CausalTransformer model):
- `src/rwm/commands/evaluate_rwm_on_rollouts.py:57` — calls
  `model(obs[:, t], act[:, t], h, c)` with `(B, WORLD_STATE_DIM)` vectors
  as history/lengths.
- `src/rwm/commands/rwm_manual_test.py:78` — passes `h_prev`/`c_prev` kwargs.
- `src/rwm/utils/behavior_memory.py:118` — calls
  `model.forward(img_t, a_t, h, c, ...)` with `(1, WORLD_STATE_DIM)` tensors.
- `src/rwm/loops/train_controller.py:205-208` — calls
  `model.forward(img, a_prev, h_prev=h, c_prev=c, ...)` with LSTM state.

All of these would raise `TypeError` or cause shape mismatches with the current
`ReducedWorldModel` which uses `CausalTransformer`.

### Stage 0 Closure Status

The contract, episode-safe split direction, deterministic probe format, and
transition regression coverage are complete. The following decisions are
approved for their specified implementation stages:

1. **Action shift in training loop**: `world_model_trainer.py:_compute_batch_loss`
   feeds `action[t-1]` while targeting `reward[t]` (generated by `action[t]`).
   Stage 2 is approved to correct this to `obs[t], action[t] → reward[t]`.
2. **Done flag ambiguity**: `collector.py` merges `terminated` and `truncated`
   into a single `done` flag. Stage 4 is approved to use no bootstrap for true
   terminations and bootstrap only valid continuations after truncation, once
   the collector stores both flags separately.
3. **Missing checkpoint**: No surviving legacy checkpoint exists for the LSTM
   path.  Stage 2 will establish a new Transformer baseline from scratch.

### Validation Command

The following command validates the current Stage 0 implementation:

```bash
cd /workspaces/tesis_v4 && python -m pytest tests/unit -v
```

Expected result: the complete current test suite passes, including the
deterministic transition-alignment and episode-safe split tests.

## Stage 0.5 — Reproducible Experiment Infrastructure

This stage improves operational structure without changing the model
architecture or the approved transition semantics. It must complete before a
new reward-prediction result is treated as experimental evidence.

### Completed Work

- **Typed configuration system** (`src/rwm/config/experiment_config.py`):
  `DataConfig`, `PerceptionConfig`, `TemporalConfig`, `ControllerConfig`,
  `TrainingConfig`, and top-level `ExperimentConfig` — frozen dataclasses
  with `to_dict()` / `from_dict()` / `to_json()` / `save()` / `load()`.
  Architectural defaults mirror current ``config.py`` constants; CLI-supplied
  run options are captured in the resolved configuration. No external
  dependencies.

- **Run directory management** (`src/rwm/utils/run_directory.py`):
  `create_run_directory()` creates ``runs/<name>/<id>/`` with
  ``config.json``, ``environment.json`` (Python, PyTorch, CUDA, platform),
  ``git_metadata.json`` (commit hash + dirty flag), and ``metrics/``,
  ``checkpoints/``, ``probes/`` subdirectories.  Deterministic ``run_id``
  when provided; timestamp-based otherwise.

- **Dataset manifest** (`src/rwm/utils/dataset_manifest.py`):
  ``build_dataset_manifest()`` records schema version, file paths with
  SHA-256 hashes and sizes, episode-safe train/val partition with split
  seed/ratio, preprocessing settings, and creation time.
  ``validate_manifest()`` checks schema version, file existence, recorded
  hashes, and train/val disjointness. Current ``.npz`` files remain supported.

- **Structured checkpoints** (`src/rwm/utils/checkpointing.py`):
  ``save_checkpoint()`` writes model state, optimizer/scheduler state,
  global step, epoch, resolved config, metrics, RNG states, and dataset
  manifest reference.  ``load_checkpoint()`` detects legacy bare
  ``state_dict`` (schema version 1 / missing) with a user warning and
  returns ``{"legacy": True, "model_state": ..., "config": None}``.

- **Seed helpers** (`src/rwm/utils/seeding.py`):
  ``set_seed(seed, deterministic=False)`` seeds Python, NumPy, and PyTorch.
  ``SeedContext`` context manager saves/restores RNG state.
  ``get_current_seed()`` / ``get_deterministic_flag()`` for metadata.

- **CLI integration** (`src/rwm/commands/train_world_model.py`):
  ``--seed`` and ``--run-id`` flags added to ``rwm train-rwm``.
  When ``--seed`` is provided, a structured run directory is created and
  the seed is applied before training begins.  Without ``--seed`` the
  command is fully backward compatible.

- **Documentation** (`docs/protocols/experiment_artifacts.md`):
  Describes all artifact formats, code usage examples, and CLI integration.

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Frozen dataclasses (not plain dicts) | Type safety, IDE support, structured equality |
| JSON serialization (not YAML/TOML) | No new dependencies, deterministic with sorted keys |
| SHA-256 for file hashes | Fast, deterministic, widely available in stdlib |
| Legacy checkpoint detection via missing ``schema_version`` | Simple, unambiguous; warning issued on load |
| ``SeedContext`` saves/restores all three RNG states | Complete isolation; no side effects after exit |
| ``--seed`` flag on ``train-rwm`` only (not all commands) | Minimal change; other commands can follow same pattern in Stage 1 |
| Map location is caller responsibility (passed to ``torch.load``) | Follows PyTorch's native API; no hidden device assumptions |

### Remaining Stage 0.5 Work (Non-Blocking)

1. **Full structured checkpoint integration in training loop.**
   ``WorldModelTrainer`` still saves a bare ``state_dict`` as
   ``best_world_model.pt``.  Integrating ``save_checkpoint()`` into the
   trainer requires changing the training loop's checkpoint path, which
   is a Stage 1 concern.

2. **Dataset manifest saved to run directory automatically.**
   The ``train_world_model`` command does not yet build and persist a
   manifest.  This will be added alongside the structured checkpoint
   integration in Stage 1.

3. **Rollout metadata (terminated/truncated, policy ID, collector config).**
   The collector schema extension is deferred to Stage 7 per the approved
   recommendations in ``docs/contracts/transition_contract.md``.

4. **CLI integration for remaining commands.**
   ``train-controller``, ``test-rwm-manually``, and ``test-rwm-rollouts``
   still lack seed/run-id flags.  These are less critical because they are
   either legacy LSTM paths (broken) or evaluation-only.

### Blocker

Do not compare experiments, claim resource efficiency, or select a baseline
until configuration, data identity, and checkpoint state are persisted.

### Validation Command

```bash
cd /workspaces/tesis_v4 && python -m pytest tests/unit/test_experiment_infrastructure.py -v
```

Expected result: 27 tests pass covering config round-trip, deterministic
serialization, run directory creation, dataset manifest creation and
validation, structured checkpoint save/load, legacy checkpoint
compatibility, seed reproducibility, and seed context isolation.

## Stage 1 — Complete the Causal Transformer Structure

### Completed Work

- **Module rename**: ``casusal_transformer.py`` → ``causal_transformer.py``.
  A deprecated import shim at the old path forwards imports with a
  ``DeprecationWarning``.

- **Structured world-model output** (`src/rwm/types.py`):
  ``WorldModelOutput`` NamedTuple replaces the anonymous 6-element tuple.
  Fields: ``world_state``, ``reward_pred``, ``mask_soft``, ``indices``,
  ``history``, ``lengths``, ``tok_mu``, ``tok_logvar``.

- **Centralized history helper** (`src/rwm/utils/history_buffer.py`):
  ``HistoryBuffer`` class with ``append()``, ``from_history()``, ``reset()``.
  No ``torch.no_grad()`` in the helper — gradient flow is preserved for
  Stage 2.

- **Active Transformer migration completed**:
  - ``ReducedWorldModel.forward()`` returns ``WorldModelOutput`` and uses
    ``HistoryBuffer`` instead of the old ``_append_token()``.
  - ``WorldModelTrainer`` updated to use named fields from
    ``WorldModelOutput``.
  - ``RolloutSimulator.warmup_state()`` and ``imagine_rollout()`` updated to
    use ``HistoryBuffer`` and ``WorldModelOutput``.
  - ``evaluate_rwm_on_rollouts.py``: marked as unavailable with clear error
    message explaining the legacy LSTM incompatibility.
  - ``rwm_manual_test.py``: marked as unavailable with clear error message.
  - ``behavior_memory.recompute_keys()``: raises ``NotImplementedError`` with
    migration guidance.
  - ``ControllerTrainer.train_on_memory()``: raises ``NotImplementedError``
    — this path will be replaced by Actor-Critic in Stage 4.

- **Stage 0.5 artifact integration**:
  - ``WorldModelTrainer`` accepts optional ``ExperimentConfig`` and
    ``dataset_manifest_ref``.  When config is provided,
    ``log_and_checkpoint()`` saves ``checkpoint_best.pt`` and
    ``checkpoint_latest.pt`` as structured checkpoints alongside the
    legacy bare ``best_world_model.pt``.
  - ``train_world_model_loop`` builds and persists a dataset manifest
    when config is provided.
  - CLI ``--seed`` flag creates a structured run directory and wires
    config through the full training pipeline.

- **Legacy preservation**:
  - Legacy modules (``WorldRNN``, ``PatchRNN``, ``WorldRNN`` tests) remain.
  - Bare ``state_dict`` checkpoints still written for backward compat.
  - The deprecation shim for ``casusal_transformer`` import is preserved.

### Transformer Validation Tests

| Test | What it verifies |
|------|-----------------|
| ``test_future_tokens_do_not_affect_earlier_output`` | Causal masking: output at position t does not depend on tokens > t |
| ``test_identical_prefix_different_suffix_gives_same_first_output`` | Output at pos 0 is unchanged when later tokens differ (length=1 extraction) |
| ``test_key_padding_mask_shape`` | Key padding mask has correct shape and True for padded positions |
| ``test_history_buffer_truncates_at_max_len`` | Buffer drops oldest tokens when context exceeds maximum |
| ``test_history_preserves_gradient_flow`` | No ``torch.no_grad()`` in HistoryBuffer |
| ``test_different_actions_produce_different_states`` | Action conditioning changes world state |
| ``test_causal_transformer_is_deterministic`` | CausalTransformer in eval mode |
| ``test_full_model_seeded_is_reproducible`` | Full model with seeded tokenizer |
| ``test_world_model_output_fields_and_shapes`` | All WorldModelOutput fields |
| ``test_history_grows_with_each_call`` | History length increases by 1 per call |
| ``test_full_vs_incremental_agree_in_eval`` | Full-sequence and incremental inference match |

### Checkpoint

- Unit tests cover causal masking, padding, truncation, determinism, shapes,
  action conditioning, and gradient flow through history.
- Full-sequence and incremental inference agree in evaluation mode within
  numeric tolerance.
- No active path references ``h_prev``, ``c_prev``, ``rnn_cell``, or
  ``world_rnn``.  Legacy LSTM paths that cannot be safely migrated raise
  clear ``NotImplementedError`` or ``typer.Exit`` with migration messages.
- Structured run artifacts, dataset manifest, and structured checkpoints
  are produced by ``rwm train-rwm`` when ``--seed`` is provided.
- Legacy bare ``state_dict`` checkpoints remain readable through
  ``load_checkpoint()``.

### Blocker

Do not build Actor-Critic on mixed Transformer/LSTM state interfaces.

## Stage 2 — Recover Full End-to-End Reward Prediction

### Completed Work

- **Corrected timing contract** (approved):

  ```
  belief b_t = Transformer(obs[t], action[t-1], history)
  Actor(b_t)             → action[t]                   (Stage 4)
  RewardHead(b_t, action[t]) → reward[t] (= r_{t+1})   (active)
  ```

  Transformer receives the **previous** action (``action[t-1]``) to
  produce a pre-action belief.  The reward head receives the **current**
  action (``action[t]``) separately.  This ensures the Actor can select
  ``action[t]`` from the belief without circular dependence.

- **ControllerTrunk** (`src/rwm/models/controller_trunk.py`):
  - ``encode(belief)`` → ``shared_repr`` (actor-ready, pre-action).
  - ``predict_reward(shared_repr, action)`` → ``reward_pred``.
  - ``forward(belief, action)`` — convenience: encode + predict_reward.
  - Reward head input: ``hidden_dim + action_dim`` (conditioned on
    current action).

- **ReducedWorldModel** API:
  - ``forward(img, prev_action, current_action, ...)`` — incremental.
  - ``forward_sequence(obs, prev_actions, current_actions, ...)`` —
    full-sequence training.  ``prev_actions[:,0]=zeros``,
    ``prev_actions[:,t]=action[t-1]``.

- **``a_prev`` renamed** to ``prev_action`` throughout the model API.
  No ambiguous naming remains — ``prev_action`` is always the action
  used for the Transformer token (``action[t-1]``).

- **Full-sequence training**: perception once per frame, one Transformer
  pass, per-step reward predictions via ``WorldModelOutput.reward_pred_seq``.
  KL aggregated from all frames.

- **Held-out validation**: ``checkpoint_best`` selected by ``val_mse``.

- **CLI output policy**: default structured dir, explicit ``--out-dir``,
  reject ``--out-dir`` + ``--run-id``.

### Transition regression tests

| Test | What it proves |
|------|----------------|
| ``test_timing_contract_prev_action_is_zeros_at_t0_and_act_tminus1_after`` | ``prev_action = zeros`` at step 0, ``action[t-1]`` at step t>0; ``current_action = action[t]`` always |
| ``test_timing_contract_reward_head_receives_current_action`` | MSE target is ``reward[:, :T]`` (reward from ``action[t]``) |

### Test suite

```
99 passed in 6.79s
```

### Validation Results

The validation script ``scripts/evaluate_reward_prediction.py`` runs four modes:

| Mode | Command | What it tests |
|------|---------|---------------|
| Smoke | ``--smoke --epochs 1`` | Infrastructure works end to end |
| Overfit | ``--overfit --epochs 50`` | Model can overfit one real episode |
| Action probe | ``--action-probe`` | Reward changes when current action changes (belief fixed) |
| Full val | (Not run, see below) | Held-out MSE vs mean-reward baseline |

#### Action-sensitivity probe (PASSED)

Holding belief fixed (same observation, zero prev_action), varying ``current_action``
produces 3 unique reward predictions:

```
action=[1.0, 0.0, 0.0] → reward = -0.0048
action=[0.0, 1.0, 0.0] → reward = -0.1974
action=[0.0, 0.0, 1.0] → reward =  0.0116
```

The reward head is properly current-action-conditioned.  **PASSED**.

### Overfit diagnostic (controlled, fixed window)

| Property | Value |
|----------|-------|
| Fixed window sequence length | 16 |
| Window reward mean / std | 0.509 / 1.766 |
| Constant-mean baseline MSE | 2.924 |
| Training loss function | ``beta=0`` (reward MSE only) |
| Optimizer steps | ~6500 (50 epochs) |
| Initial reward MSE | ~1.0 |
| Final reward MSE | ~0.0004 |
| MSE reduction (vs baseline) | **4 orders of magnitude below baseline** |

**Interpretation: the model can clearly overfit.** The reward-MSE-only diagnostic
(beta=0, no KL, no observational dropout, fixed window) reduces the loss from
~1.0 to ~0.0004, which is 7310× below the constant-mean baseline of 2.924.  This
confirms that the entire gradient pipeline (perception → Transformer →
controller trunk → reward head) is functional.  No NaN or gradient explosion
was observed.

#### Runtime profile (overfit diagnostic)

| Phase | Throughput |
|-------|-----------|
| ``forward_sequence`` (16 frames, batch 1) | ~16 batches/s |
| Evaluation (same data, same batch size) | ~5 batches/s |

The evaluation is slower because it iterates the full validation loader with
``forward_sequence``, which runs perception once per frame.  For a single
window this is negligible; for 1065 windows it becomes a bottleneck.

#### Held-out validation (BOUNDED)

A bounded held-out run was not completed because the evaluation step over the
validation set dominates runtime.  The test ``test_baseline_mse_uses_training_set_mean``
validates the metric aggregation logic: the baseline MSE is computed using
the actual training-set reward mean and aggregated over all validation batches.
This test passes.

**Recommendation**: Before a full validation run, the evaluation should be
profiled to determine whether the bottleneck is perception (per-frame encoder
forward) or data loading.  If it is perception, the validation set should be
subsampled to a fixed number of windows for consistent timing.

### Follow-up Optimisations and Ablations

These are not prerequisites for the controlled reward-only overfit checkpoint.
They become priorities after a bounded held-out reward baseline is established,
and before making efficiency claims.

1. **Top-K straight-through mask integration.** The selector builds a
   hard-forward/soft-backward mask, but the active spatial head gathers hard
   indices and does not consume that mask. Selected tokens receive direct
   reward gradients; nonselected token values do not receive a per-step soft
   routing gradient. Preserve hard Top-K as the baseline; add an optional
   training-only straight-through masked pooling path only if validation shows
   it is needed. Evaluation remains deterministic hard Top-K.

2. **Tokenizer evaluation policy.** The variational tokenizer samples noise
   even in evaluation mode. Add and compare a deterministic evaluation option
   that uses token mean `mu`, recording the policy in each experiment.

3. **Pruning-strength ablation.** Compare `K=8`, `K=16`, `K=32`, and a
   high-K reference on reward quality, token diversity, latency, throughput,
   and memory. K=8 is intentionally aggressive: 8 of 225 overlapping tokens.

4. **Perception batching and profiling.** `forward_sequence()` still runs
   perception in a Python timestep loop. Profile data loading, perception,
   spatial pooling, and Transformer time. If perception dominates, batch
   frames as `(B*T, C, H, W)` and restore sequence shape with an equivalence
   and gradient regression test.

5. **Forced observational-dropout evaluation.** Add an explicit deterministic
   missing-observation evaluation mode only after reward validation is stable.
   Compare increasing dropout horizons; do not infer robustness from current
   `force_keep_input=True` reward training.

#### Result table template

For future Stage 2 training with non-random data:

```
+----------------+----------+----------+----------+
| Metric         | Epoch  1 | Epoch 10 | Epoch 20 |
+----------------+----------+----------+----------+
| Train MSE      |          |          |          |
| Val MSE        |          |          |          |
| Val MAE        |          |          |          |
| Baseline MSE   |          |          |          |
| Throughput     |          |          |          |
+----------------+----------+----------+----------+
| gn_encoder     |          |          |          |
| gn_tokenizer   |          |          |          |
| gn_scorer      |          |          |          |
| gn_selector    |          |          |          |
| gn_spatial_hd  |          |          |          |
| gn_world_hd    |          |          |          |
| gn_controller  |          |          |          |
+----------------+----------+----------+----------+
```

#### Dataset limitations

| Property | Value |
|----------|-------|
| Total episodes | 20 |
| Total steps | 5339 |
| Episode length range | 148–845 (mean 267) |
| Reward mean / std | 0.078 / 0.830 |
| Reward range | -0.10 to 10.81 |
| Done flags | 0 (all idle-stopped) |
| Policies | 10 random, 10 random_smooth |
| Scenarios | 1 (scenario_0 only) |
| Action zeros | 0 / 5339 (all non-zero rewards) |

**Critical limitation**: all data was collected under random or random-smooth
policies.  Reward peaks (max 10.81) are rare.  The pixel-to-reward mapping is
extremely sparse and noisy.  Future training should use either:
- A partially trained controller (from Stage 4), or
- Human-collected demonstration data, or
- Dramatically more random data (100k+ frames).

#### Retained artifacts

Transient smoke runs are intentionally not retained. Reproducible anchor
artifacts (configuration, environment, manifest, metrics, structured
checkpoint, and fixed probe set) are retained under
``runs/component_refinement/00_reward_anchor_pre_kl_fix/seed_42/`` and
``seed_43/``. The convention is recorded in
``runs/component_refinement/RUN_INDEX.md``.

### Stage 2 Status

**Initial end-to-end anchor passed; research claims remain gated by Stage 2.5.**

The timing contract and reward pipeline are connected and can overfit a
nonconstant sequence. Two beta-zero checkpoints beat a constant-mean held-out
baseline by 14--16 percent. This is sufficient evidence to preserve the model
as an experimental anchor, but the later audit found that overlapping windows,
startup/early-push effects, sparse random-policy rewards, and a weak baseline
limit the conclusion.

The current evidence does **not** yet establish variational token similarity,
benefit from learned Top-K, action-conditioned masked dynamics, or readiness for
policy training. Stage 3 is therefore no longer authorized solely by the
constant-mean comparison.

## Stage 2.5 — Validate the Architecture Hypotheses

Detailed definitions, metrics, and gates are in
`docs/plans/architecture_validation_plan.md`.

### A — Measurement foundation

- Build seeded, immutable development/validation/test CarRacing scenarios.
- Add competent human/heuristic trajectories and same-state branched action
  sequences.
- Evaluate each transition once with reward-event, horizon, startup, and HUD
  strata plus stronger baselines.

### B — Correct and accelerate the anchor

**B.1 — KL reduction (COMPLETE).**
The variational-token KL is now computed per posterior element before any
reduction.  The exact contract is:

```python
kl_per_element = 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
kl = kl_per_element.mean()      # mean over (B, T, P, D)
```

``forward_sequence`` preserves the full ``(B, T, P, D)`` posterior shape
instead of averaging over time before applying the nonlinear KL formula.
The reusable helper ``kl_normal()`` in ``world_model_trainer.py`` implements
this contract and is tested with known-posterior analytical values, a
regression case proving that pre-averaging gives a different result, and
gradient flow verification.

| Test | Result |
|------|--------|
| ``kl_normal(mu=0, logvar=0) = 0`` | PASSED |
| ``kl_normal(mu=1, logvar=0) = 0.5`` | PASSED |
| Per-element KL > pre-averaged KL for non-constant posteriors | PASSED |
| Gradients flow through KL with beta>0 | PASSED |
| Changing one timestep changes KL | PASSED |
| Full test suite with smoke training | 153 passed |

**B.2 — Top-K straight-through gradients (COMPLETE).**
The straight-through selection mask now flows into ``SpatialAttentionHead``
instead of being discarded before pooling. The soft surrogate has total mass
``K``, not sum-one. Training computes dense patch values for the surrogate so
unselected scorer logits receive finite nonzero gradient; evaluation retains
the original hard K-token value projection and pooling path. The training-only
value projection expands from K=8 to N=225 tokens (28.1x for this small
projection), so measured end-to-end throughput remains a later checkpoint;
there is no all-token inference projection cost. See
``docs/plans/architecture_validation_plan.md`` for the exact equations.

| Test | Result |
|------|--------|
| Eval forward equals legacy hard gather | PASSED |
| Eval projects only K selected values | PASSED |
| Non-selected tokens do not affect eval output | PASSED |
| Unselected logits receive dense gradient | PASSED |
| Eval is deterministic | PASSED |
| Training has Gumbel-based variation | PASSED |
| Model integration (shapes, trace parity) | PASSED |

**B.3 — Sliding-window previous-action boundary (COMPLETE).**
The predecessor action for each window is now the actual ``action[offset-1]``
for mid-episode windows, rather than always zeros.  True episode starts
(``offset == 0``) still use zeros.  ``RolloutDataset`` returns the new
``predecessor_action`` field; the trainer uses it for ``prev_actions[:, 0]``.
No predecessor crosses an episode boundary.

| Test | Result |
|------|--------|
| Mid-episode window gets correct predecessor | PASSED |
| Episode-start window gets zeros | PASSED |
| Predecessor does not cross episode boundary | PASSED |
| Spy test proves prev_action at t=0 uses batch predecessor | PASSED |
| Trainer rejects a batch missing predecessor metadata | PASSED |
| Full test suite | 166 passed |

**Performance impact:** One additional ``(A,)`` tensor per sample — negligible.
No model inference overhead.
**B.4 — Corrected reward-prediction anchor experiment (COMPLETE).**
A matched beta sweep (0.0, 0.01, 0.1 × seeds 42, 43) was run after all B.1–B.3
fixes.  Results are in ``runs/component_refinement/01_corrected_reward_anchor/``.

| Beta | Seed 42 ratio | Seed 43 ratio |
|------|---------------|---------------|
| 0.00 | 0.857         | 0.812         |
| 0.01 | **0.828**     | 0.837         |
| 0.10 | 0.865         | **0.791**     |

All 6 runs beat the constant-mean baseline. Beta-weighted runs are competitive
with beta=0.0, confirming that the corrected KL (B.1) no longer dominates the
loss. ``beta=0.10`` has the best two-seed mean MSE/ratio, while ``beta=0.01``
has the best worst-seed ratio and lowest two-seed variation. ``beta=0.10`` is
the default for subsequent experiments: it retains stronger variational
pressure while remaining reward-competitive. This is a deliberate research
configuration, not a demonstrated universal optimum. Action-conditioning probe
passes for all 6 checkpoints (4/4 unique predictions).

- Compare a small nonlinear state-action reward head with the current additive
  linear head.
**B.5 — Vectorised B*T perception and controller (COMPLETE).**
``forward_sequence`` now reshapes ``(B, T, ...)`` frames into ``(B*T, ...)``
and runs encoder, tokenizer, scorer, selector, and spatial head in a single
batched call.  Controller reward prediction is similarly vectorised over
``(B*T)`` positions.  The causal Transformer and all other components are
unchanged.

| Metric | Before (per-frame) | After (B*T) | Speedup |
|--------|-------------------|-------------|---------|
| ms/step | 68.7 | 25.4 | **2.71x** |
| windows/sec | 116.4 | 315.5 | **2.71x** |
| Peak GPU (GB) | 0.272 | 0.561 | 2.06x |

Eval-mode incremental/sequence parity holds exactly.  Full test suite passes.
The benchmark measures preloaded synthetic compute only. A real end-to-end
retrain must be timed separately because NPZ decompression, PIL transforms,
and host-to-device loading are not included. The previously observed 10-epoch
anchor runs took roughly 9 minutes per seed, not 90 minutes.

### C — Perception ablations

**C.0 — Configurable reward-head architecture (COMPLETE).**
The reward head in ``ControllerTrunk`` is now configurable:

- ``"linear"`` (default, legacy compatible):
  ``reward = Linear(concat(c_t, action_t))``
- ``"nonlinear"``:
  ``reward = Linear(concat(c_t, action_t), hidden) → ReLU → Linear(hidden, 1)``

``ControllerConfig`` records ``reward_head_kind`` and ``reward_head_hidden_dim``.
The default nonlinear experimental width is 32; it is ignored by the default
linear head.
Structured checkpoints store this metadata; loading a checkpoint without it
defaults to ``"linear"``.  ``model_from_checkpoint()`` in ``checkpointing.py``
builds the correct architecture from the checkpoint config.

| Test | Result |
|------|--------|
| Linear forward shapes | PASSED |
| Nonlinear forward shapes | PASSED |
| Gradients flow to belief and action (nonlinear) | PASSED |
| Legacy bare state_dict loads as linear | PASSED |
| Structured nonlinear checkpoint round-trip | PASSED |
| Config JSON round-trip | PASSED |
| Full test suite | 183 passed |

**Compatibility contract:** ``model_from_checkpoint()`` reads
``config.controller.reward_head_kind`` from the checkpoint; if absent (legacy
checkpoint or ``config=None``), defaults to ``"linear"``.  No silent guessing
from state-dict keys.

**C.1 — Reward-head capacity ablation (COMPLETE).**
Two-seed comparison (42, 43) of linear vs nonlinear (83→32→1, ReLU) reward
head at beta=0.1.  Results in ``runs/component_refinement/03_nonlinear_reward_head/``.

| Seed | Head | Best val MSE | Ratio | vs baseline |
|------|------|-------------|-------|-------------|
| 42   | linear | 0.4663 | 0.828 | 17.2% below |
| 42   | nonlinear | 0.4885 | 0.867 | 13.3% below |
| 43   | linear | 0.4952 | 0.787 | 21.3% below |
| 43   | nonlinear | 0.5054 | 0.803 | 19.7% below |

**Verdict:** The nonlinear head does not consistently beat the linear baseline.
Linear wins on seed 42; seed 43 is essentially tied.  The linear head remains
the default for subsequent experiments.  Action probes: 4/4 unique for all 4
checkpoints.

**C.2A — Top-K selection ablation (COMPLETE).**
Four runs (fixed_uniform × 2 seeds + fixed_random × 2 seeds) vs learned
(Stage 02 anchors) at K=8, beta=0.1. The static controls used cached frames;
the historical learned anchors did not, so this stage compares reward quality,
not runtime.

| Selection | Mean ratio | Seed 42 | Seed 43 |
|-----------|-----------|---------|---------|
| learned   | **0.808** | 0.828   | 0.787   |
| fixed_uniform | 0.868 | 0.865 | 0.871 |
| fixed_random  | 0.834 | 0.867 | 0.800 |

**Verdict:** Learned adaptive Top-K consistently beats both static controls
across both seeds.  The learned selector provides better held-out reward
prediction than fixed patch-position candidates.  All 6 checkpoints pass
the action probe (4/4).  See ``runs/component_refinement/RUN_INDEX.md`` for
the full table.

**C.2B — K-ablation (COMPLETE).**
Eight cached runs at K=4/8/16/32 × seeds 42/43 (beta=0.1, linear reward head,
10 epochs).  K=8 was re-run under the same cached conditions alongside K=4/16/32.

| K   | Seed | Ratio | Mean |
|-----|------|-------|------|
| 4   | 42   | 0.931 | 0.886 |
| 4   | 43   | 0.841 |      |
| 8   | 42   | 0.845 | 0.806 |
| 8   | 43   | 0.767 |      |
| 16  | 42   | 0.834 | 0.822 |
| 16  | 43   | 0.809 |      |
| 32  | 42   | 0.896 | 0.870 |
| 32  | 43   | 0.843 |      |

**Verdict:** K=8 and K=16 are close (mean ratio 0.806 vs 0.822), while both
clearly outperform K=4 (0.886) and K=32 (0.870).  K=8 remains the default
because it gives the best observed ratio while preserving the intended tighter
spatial-information bottleneck.  The current dense implementation has the same
dominant CNN/tokenizer/scorer FLOPs for every K; K=8 leaves the strongest path
to later sparse-inference savings.  See
``runs/component_refinement/RUN_INDEX.md`` for full details.

**C.3 — Tokenizer evaluation policy (COMPLETE).**

| Checkpoint | Policy | Mean ratio | Ratio σ | mean reproducible? | 4/4 probe |
|-----------|--------|:----------:|:-------:|:------------------:|:---------:|
| K=8 seed 42 | mean | **0.84529** | — | yes, bitwise | yes |
| K=8 seed 42 | sample | 0.84768 | 0.00117 | — | yes |
| K=8 seed 43 | mean | **0.76662** | — | yes, bitwise | yes |
| K=8 seed 43 | sample | 0.77076 | 0.00371 | — | yes |

**Verdict:** The mean policy is deterministic (bitwise identical across
inference RNG seeds) and marginally better (lower MSE) than sample for both
seeds.  Sample variance is low (σ ≤ 0.004 in ratio).  See
``runs/component_refinement/06_tokenizer_eval_policy/RESULTS.md`` for full
details.  ``mean`` becomes the default for future evaluation experiments.

### D.0 — Masked factual evaluator (COMPLETE)

The masked dynamics interface is a deterministic, testable factual
observation-masking evaluator.  It is not imagination training.

**Canonical semantics:**

- Tokenizer evaluation mode ``mean``.
- ``forward_sequence`` accepts an optional ``observation_keep: BoolTensor (B, T)``.
  ``True`` = use spatial representation from image; ``False`` = replace spatial
  representation with zeros.
- ``forward()`` accepts an optional ``observation_keep: bool`` per step.
- All-True ``observation_keep`` matches the existing visible output bitwise.
- Actions remain explicit: ``token[t]`` uses previous action ``a[t-1]``;
  reward prediction uses current action ``a[t]``.
- The model receives no image-derived information during the contiguous masked
  horizon, but it may use causal history and action history; factual images
  become visible again after that horizon.
- Perception (encoder → tokenizer → scorer → selector → spatial head) still
  runs on masked frames for diagnostic transparency; only the spatial output
  is zeroed.

**Reusable evaluator** ``MaskedFactualEvaluator`` under
``src/rwm/evaluation/masked_factual_evaluator.py``:

- Takes a model, loader, warmup, mask horizon, and action variant.
- Reports per horizon: transition count, MSE, MAE, baseline MSE, ratio,
  visible-reference MSE, delta from visible, and policy/config provenance.
- Three action-history variants applied only from the masked boundary onward
  (the visible warmup always uses factual history):
  a) **correct** — factual previous actions (``prev[:,0]=predecessor``)
  b) **zero** — all-zero previous actions
  c) **shifted** — ``prev[t] = actions[t]`` (uses current action as prev)
- Current action into the reward head remains factual in all variants.

**Tests (20 focused):**

- Action-variant indexing contract (correct, zero, shifted).
- Observation mask construction: warmup visible, masked horizon, clip to T.
- All-visible mask equals existing visible ``forward_sequence`` output.
- Fully masked output does not depend on image content.
- Warmup image changes do affect masked outputs.
- Masked-zone image changes do not affect masked outputs.
- Outputs have finite values and preserve gradients.
- Incremental ``forward()`` with ``observation_keep=False`` zeros spatial rep.
- ``MaskedFactualEvaluator`` returns per-horizon results with finite metrics.
- Legacy unmasked ``forward_sequence``/``forward`` unchanged.

**Validation results** (see ``runs/component_refinement/07_masked_factual_dynamics/RESULTS.md``):

- All masked ratios > 1.0 — expected; these anchors were trained with all
  observations visible.  This is an interface sanity check, not evidence of
  learned blind dynamics.
- MSE degrades with shorter horizon (more masking).
- ``correct`` variant dominates ``zero`` and ``shifted`` — factual previous-action
  history provides useful temporal context even without visual input.
- ``zero`` (no action history) is consistently worst.
- Seed 43 has larger masked losses under this protocol; this is descriptive,
  not evidence for a causal attribution about visual reliance.
- Aggregate delta from visible decreases with longer horizons, but those
  horizons score different later transition sets; it is not evidence that more
  blind context helps.

**Important caveat:** The existing reward anchors (C.0–C.2B) were trained with
all observations visible.  Their masked results are a structural sanity check
of the interface, not evidence of learned blind dynamics, imagination ability,
or policy readiness.  No training or fine-tuning has been performed for masked
operation; the model has never seen masked training examples.

**Unit coverage:** 22 focused tests in ``tests/unit/test_masked_dynamics.py``
plus CLI import tests.  Full suite: 291 passed.

### D.1 — Temporal observational-dropout training (COMPLETE)

**Training results** (``runs/component_refinement/08_masked_reward_anchor/``):

| Model | Seed 42 visible ratio | Seed 43 visible ratio |
|-------|:---------------------:|:---------------------:|
| Frozen visible-only (Stage 2.5C) | 0.845 | 0.767 |
| Masked-trained | 0.882 | 0.853 |

**Masked factual reward prediction (correct action history):**

| Seed | Frozen ratio range | Masked-trained ratio range | All horizons improve? |
|:----:|:------------------:|:--------------------------:|:--------------------:|
| 42 | 1.076–1.338 | **0.928–0.954** | YES (15/15) |
| 43 | 1.209–2.824 | **0.865–0.925** | YES (15/15) |

Every single masked evaluation (3 variants × 5 horizons × 2 seeds = 30
comparisons) improves relative to the frozen visible-only anchor.  Visible
reward quality degrades moderately (seed 42: +0.037; seed 43: +0.086) but
remains below 1.0 and passes the 4/4 action probe.

**Verdict:** The temporal mask curriculum successfully teaches masked reward
prediction.  The D.1 gate is satisfied.  Stage 3 can proceed with the
masked-trained anchor for its imagination interface.

### Stage 2.5 checkpoint

- The corrected anchor matches or exceeds the current reward result on the
  fixed suite.
- Retained perception mechanisms beat declared matched baselines across seeds.
- D.1 demonstrates improved masked factual reward prediction relative to the
  visible-only anchor, without collapsing visible reward quality.
- Runtime, MACs, latency, memory, and unique environment frames are reported.

Only then proceed to general open-loop imagination and Actor-Critic training.

## Stage 3 — Differentiable Bounded-Context Imagination Interface (COMPLETE)

Stage 2.5D.1 is the experimental gate for masked dynamics. Stage 3 turns the
validated observed/masked transition semantics into a reusable, differentiable
imagination interface for Actor-Critic training (Stage 4).

### State semantics (implemented)

At time ``t``, the causal state contains raw token history ``H_t`` through::

    token[t] = cat(spatial_rep(obs[t]), action[t-1])
    z_t = Transformer(H_t)                          — belief
    T ̂_hat[t] = Reward(z_t, action[t])               — score
    H_{t+1} = append(H_t, cat(zero_spatial, action[t]))  — advance

Imagined steps have no image input.  The spatial representation is replaced
with zeros; the action from the previous step provides the temporal context.

### Implementation

File ``src/rwm/imagination.py``:

**ImaginationRollout** (``nn.Module``):

- ``warmup(obs, prev_actions, current_actions) → ImaginationState``:
  Runs ``forward_sequence`` for vectorised perception and extracts per-step
  causal beliefs via a re-entrant ``world_hd`` call. Returns ``history``
  (``B, T, V+A``) and ``beliefs`` (``B, T, D``).

- ``score(belief, action) → (B, 1)``:
  ``ControllerTrunk.encode + predict_reward``.  Pure tensor-in/tensor-out.

- ``advance(history, lengths, action) → (new_history, new_lengths, new_belief)``:
  Builds blind token ``cat(zeros, action)``, appends via ``HistoryBuffer``
  (left-truncation preserved), runs ``world_hd``.  No perception/CNN called.

- ``rollout(history, lengths, initial_belief, actions (B, H, A)) → RolloutOutput``:
  Per-step score-then-advance loop.  Returns ``states (B, H, D)``,
  ``rewards (B, H)``, ``next_state (B, D)``, accumulated ``history`` and
  ``lengths``.  Gradient-preserving throughout.

**Data containers:**

- ``ImaginationState``: ``history``, ``lengths``, ``beliefs``, ``current_belief``.
- ``RolloutOutput``: ``states``, ``actions``, ``rewards``, ``next_state``,
  ``history``, ``lengths``.

**Design decisions:**

- Never uses ``torch.no_grad()``, ``.item()``, or ``.detach()`` in trainable
  paths.  Gradients flow from imagined rewards through to action tensors and
  world-model parameters.
- Uses ``HistoryBuffer`` for canonical bounded-context truncation (no
  duplication of truncation rules).
- The inactive ``RolloutSimulator`` draft is preserved for LSTM comparison
  but is NOT used by any active code path.

### Tests (21 tests in ``tests/unit/test_imagination.py``)

1. **Warmup matches forward_sequence** — history tokens and beliefs identical.
2. **Warmup matches incremental forward** — step-by-step ``model.forward()``
   produces the same history as a single ``warmup()`` call.
3. **Advance matches D.0 masked** — a single blind advance reproduces the
   first masked-position belief from ``forward_sequence`` with
   ``observation_keep`` masking.
4. **Multi-step shapes and truncation** — correct tensor shapes, history
   truncated at ``SEQ_LEN``, warmup prefix preserved, score order proved.
5. **Score-then-advance timing spy** — first score uses the initial belief
   and the correct action; second belief differs after advance.
6. **Gradient flow** — gradients from summed imagined rewards reach both
   the input action tensor and the world model's transformer/controller
   parameters.  Warmup history does not block gradients.
7. **No future observation** — imagined tokens have zero spatial
   representation; rewards are deterministic given the same actions.
8. **Existing APIs unchanged** — ``forward_sequence`` and incremental
   ``forward`` still work identically.

**Full suite: 309 passed** (21 new imagination tests, no regressions).

### Checkpoint (all satisfied)

- [x] Incremental observed steps and the interface's observed mode agree with
  the validated factual model.
- [x] The interface reproduces the fixed masked-horizon evaluator's predictions.
- [x] Rollouts remain tensors with finite gradients and correct action/reward
  phase alignment.
- [x] Deterministic evaluation is reproducible.
- [x] No future observation can affect imagined states/rewards.
- [x] Existing visible reward APIs unchanged.

### Blocker resolved

Stage 2.5D.1 produced action-sensitive masked dynamics.  This implementation
reproduces those results exactly.  The gate for Stage 4 is open.

### Next: Stage 4 — Actor-Critic Head Calibration

## Stage 4 — Actor-Critic Head Calibration (Frozen World Model) (COMPLETE)

### New files

- ``src/rwm/distributions.py`` — ``BoundedGaussian`` distribution
- ``src/rwm/models/actor_critic.py`` — Actor, Critic, ActorCritic, losses
- ``tests/unit/test_actor_critic.py`` — 36 tests

### Config

``ActorCriticConfig`` added to ``src/rwm/config/experiment_config.py``
(``frozen=True``, serializable).  Fields: ``hidden_dim=64``, ``actor_lr=3e-4``,
``critic_lr=3e-4``, ``gamma=0.997``, ``lambda_=0.95``, ``entropy_coef=1e-3``,
``target_update_rate=0.01``.

### Distribution: ``BoundedGaussian``

- Dim 0 (steering): ``tanh`` → ``[-1, 1]``.
- Dims 1, 2 (gas, brake): ``sigmoid`` → ``[0, 1]``.
- ``rsample()`` — reparameterized (gradients flow through).
- ``mode()`` — deterministic (squashed mean).
- ``log_prob(action)`` — inverse transform + change-of-variables correction.
- ``entropy()`` — raw Gaussian approximation.

### Actor head: ``80 → 64 → (mean, logstd)`` for 3 actions.

One hidden ReLU layer.  ``logstd`` clamped to ``[-10, 2]``.

### Critic head: ``80 → 64 → scalar V(c_t)``.

One hidden ReLU layer.  Output is ``(B, 1)``.

### Combined module: ``ActorCritic``

Freezes the full ``ReducedWorldModel`` (including ``ControllerTrunk``).
Owns ``actor``, ``critic``, ``target_critic`` (Polyak-updated).  Two
separate Adam optimisers for actor and critic.

**Key method ``optimizer_step``**:

1. ``c_t = encode(z_t).detach()`` — world-model gradient block.
2. ``dist = actor(c_t)``, ``values = critic(c_t)``.
3. ``target_values = target_critic(c_t)`` (``no_grad``).
4. ``advantages = TD(λ=0) residual``.
5. Actor loss: ``-mean(stopgrad(adv) * log_prob) - entropy_coef * entropy``.
6. Critic loss: ``MSE(values, λ-returns)``.
7. Backprop each, step optimisers, Polyak-update target.

### λ-return computation

Backward recurrence::

    G_t = r_t + γ * continuation_t *
          ((1 - λ) * V_target(s_{t+1}) + λ * G_{t+1})

where ``V_target(s_{t+1}) = bootstrap_values[:, t+1]`` for ``t < T-1``,
and ``V_target(s_T) = bootstrap_value`` (a separate ``(B,)`` argument).

### Termination / continuation contract

Two boolean flags per step:

- ``terminated[t] = True``
    True environment terminal state.  ``continuation[t]`` must be False.
    Return target is ``r_t`` — no bootstrap.

- ``terminated[t] = False, continuation[t] = True``
    A valid next state exists (truncation or continuation).  Standard λ-return
    bootstrap applies.

- ``terminated[t] = False, continuation[t] = False``
    An explicit non-bootstrap boundary. Return target is ``r_t``. This is
    not the normal imagined-horizon case: a rollout produced by Stage 3 has
    a valid final latent state and therefore uses ``continuation=True`` plus
    its target-critic ``bootstrap_value``.

- ``terminated[t] = True, continuation[t] = True``
    **Invalid** — raises ``ValueError`` with position index.

``bootstrap_value (B,)`` provides ``V_target(s_T)``, used only when
``continuation[:, -1]`` is True.

### Entropy

``BoundedGaussian.entropy()`` — raw-Gaussian approximation (lower bound,
cheap, documented as such).

``BoundedGaussian.entropy_sample()`` — single-sample Monte-Carlo estimate
through the full squashed distribution.  Has finite gradients and is used
by the actor loss.  The distinction is documented in code and tested.

### Tests (42 tests, all pass)

1. **Action bounds** — ``rsample`` and ``mode`` respect ``[-1, 1]`` /
   ``[0, 1]``.
2. **Deterministic repeatability** — ``mode`` reproducible; ``rsample``
   has valid gradients.
3. **log_prob finite near bounds** — finite and in plausible range.
4. **λ-returns with new contract** — 7 scenarios: no-discount, single-step
   bootstrap, λ=0/1, terminated mid-sequence, truncated final bootstrap,
   explicit no-bootstrap boundary, imagined-horizon final bootstrap,
   hand-calculated two-step.
5. **Termination/continuation semantics** — terminated steps produce no
   bootstrap; truncated steps use supplied bootstrap value; imagined-
   horizon uses the final latent target value when available; an explicitly
   non-bootstrap boundary is reward-only.
6. **Critic overfits** — synthetic return dataset memorised.
7. **Advantage direction** — positive → increased log-prob; negative →
   decreased log-prob.
8. **Entropy direction** — higher entropy coefficient → higher entropy.
9. **Target Critic Polyak update** — hard initial copy + blending verified.
10. **Freeze contract** — world-model params bitwise unchanged after 10
    optimizer steps.
11. **Validation** — ``terminated=True, continuation=True`` raises
    ``ValueError``; non-bool dtypes raise ``TypeError``; shape mismatches
    raise ``ValueError``; propagates through ``optimizer_step``,
    ``compute_lambda_returns``, and ``compute_td_advantage``.
12. **entropy_sample** — finite, has gradients, differs from raw
    approximation, non-deterministic across calls.
13. **Config round-trip** — defaults, serialisation, frozen.

### Full suite: 345 passed (36 new, no regressions).

### Checkpoint (all satisfied)

- [x] Distribution samples and deterministic actions respect CarRacing bounds.
- [x] λ-returns match hand-calculated trajectories including termination.
- [x] Critic overfits known synthetic returns.
- [x] Positive and negative advantages move action likelihood in the correct
  direction.
- [x] Actor and Critic optimise without changing world-model parameters.

### Blocker resolved

Mechanical calibration is complete.  Policy and value learning are verified
with correct inputs.  Stage 5 (frozen-imagination training) can proceed.

## Stage 5 — Train Actor-Critic in Imagination (Frozen Model) (COMPLETE — implementation, validation pending long training)

### New files

- ``src/rwm/trainers/imagined_actor_critic.py`` — ``ImaginedACTrainer`` loop
- ``scripts/train_imagined_actor_critic.py`` — thin CLI with ``--smoke``
- ``tests/unit/test_imagined_actor_critic.py`` — 20 tests

### Training architecture

::

    batch → warmup (4 frames) → z_0
    for h in 0…H-1:
        c_h = ControllerTrunk.encode(z_h).detach()
        dist = Actor(c_h)
        a_h = dist.rsample()
        r_h = RewardHead(z_h, a_h)          # frozen, no_grad
        z_{h+1} = advance(z_h, a_h)         # blind, frozen
    V_h = Critic(c_h)                       # online
    target_V_h = TargetCritic(c_h)          # target (Polyak)
    bootstrap = TargetCritic(encode(z_H))   # final next-state value
    λ-returns, TD advantages → ActorCritic.optimizer_step

### Termination / continuation contract (imagined rollouts)

``terminated`` is always ``False`` and ``continuation`` is always ``True``
for every imagined step, including the last.  The ``bootstrap_value``
comes from the target Critic evaluated on the final imagined state ``z_H``.

### Key design decisions

- **Frozen world model**: ``model.eval()``, ``requires_grad_(False)`` on all
  parameters.  Verified via hash parity before/after training.
- **Imagination horizon**: configurable in ``[1, 12]``; default ``H=4``.
  Initial training must sample only short horizons ``{1, 2, 4}``; ``H=8``
  requires a separate stability gate and ``H=12`` is initially stress-only.
- **Warmup**: 4 observed frames from dataset windows, using the correct
  ``prev_actions`` / ``current_actions`` timing contract.
- **Actor objective**: score-function policy gradient with ``entropy_sample()``.
  Actions detached for the log-prob to avoid dual-path gradients through the
  same Actor parameters.
- **Critic objective**: MSE against λ-returns using the Stage-4 contract.
- **Target Critic**: Polyak update after each critic step.
- **Persistence**: metrics CSV, JSON config, structured AC checkpoint,
  anchor provenance file.

### Tests (20 tests, all pass)

1. **Action timing** — action scored = action advanced; actions not from
   future observations.
2. **No future observation** — imagined tokens have zero spatial component.
3. **Horizon validation** — ``H ≤ 12``; ``H > 12`` and ``warmup < 1`` raise.
4. **Final bootstrap** — target Critic on ``z_H`` produces correct value.
5. **Params change** — Actor and Critic weights change after 3 steps.
6. **Frozen WM** — all world-model params bitwise identical after 5 steps;
   ``requires_grad`` all False.
7. **Action bounds** — all actions satisfy ``[-1, 1]`` / ``[0, 1]``.
8. **Finite gradients** — losses and grads finite; target Critic changes.
9. **Reproducibility** — same seed + model gives identical first-step metrics.
10. **Checkpoint round-trip** — save/restore AC state dict; anchor info and
    metrics CSV written.

### Smoke gate output (B=2, H=4, 10 steps, seed42 masked anchor)

::

    [    1] actor_loss=-0.0117 | critic_loss=0.0165 | entropy=0.2288 | value_mean=0.0720 | imagined_reward_mean=-0.0536 | action_mean=0.3419 | action_std=0.3444  |  0.7s
    [    2] actor_loss=-0.0114 | critic_loss=0.0410 | entropy=0.3183 | value_mean=0.0765 | imagined_reward_mean=0.0429 | action_mean=0.3241 | action_std=0.4322  |  0.8s
    [    3] actor_loss=0.0833  | critic_loss=0.2703 | entropy=0.7354 | value_mean=0.0863 | imagined_reward_mean=0.2140 | action_mean=0.2256 | action_std=0.4333  |  0.8s
    [    4] actor_loss=-0.0114 | critic_loss=0.2137 | entropy=-0.1469| value_mean=0.0857 | imagined_reward_mean=0.1326 | action_mean=0.1942 | action_std=0.5621  |  0.8s
    [    5] actor_loss=-0.0048 | critic_loss=0.0200 | entropy=0.6825 | value_mean=0.0654 | imagined_reward_mean=-0.0626 | action_mean=0.4303 | action_std=0.3404  |  0.8s
    [    6] actor_loss=0.0065  | critic_loss=0.0212 | entropy=0.3361 | value_mean=0.0697 | imagined_reward_mean=-0.0549 | action_mean=0.3027 | action_std=0.5375  |  0.8s
    [    7] actor_loss=0.0130  | critic_loss=0.0908 | entropy=0.4124 | value_mean=0.0971 | imagined_reward_mean=0.1371 | action_mean=0.3074 | action_std=0.4740  |  0.9s
    [    8] actor_loss=-0.0098 | critic_loss=0.0224 | entropy=-0.1193| value_mean=0.0608 | imagined_reward_mean=-0.0555 | action_mean=0.4762 | action_std=0.2467  |  0.9s
    [    9] actor_loss=0.0146  | critic_loss=0.0310 | entropy=0.0416 | value_mean=0.0551 | imagined_reward_mean=-0.0784 | action_mean=0.1792 | action_std=0.5294  |  0.9s
    [   10] actor_loss=-0.0362 | critic_loss=0.6992 | entropy=-0.1915| value_mean=0.1240 | imagined_reward_mean=0.2940 | action_mean=0.1398 | action_std=0.5694  |  0.9s

**Frozen WM parity**: hash ``240f0dd4d0328400`` identical before and after
10 training steps.

### Full test suite: 373 passed (20 new, 0 regressions)

### Checkpoint (all satisfied — unit level)

- [x] Short differentiable imagined trajectories generated from observed warmup.
- [x] Actor gradient uses score-function estimator (likelihood ratio) with
      detached advantages.
- [x] Critic trained with λ-returns against target Critic.
- [x] World model frozen throughout — hash-verified, ``requires_grad``-asserted.
- [x] Imagined horizon ≤ 12 enforced.
- [x] Reproducible with fixed seed + mean tokenizer.
- [x] Structured AC checkpoint round-trip works; anchor provenance recorded.

### Remaining (long training / real evaluation — not yet run)

- Predicted return improvement across seeds and steps.
- Actor entropy not collapsing.
- Critic error bounded as policy changes.
- Real-environment evaluation after sufficient training.

### Current blocker

Long training and environment evaluation have not been run.  The
implementation is mechanically correct (verified by 20 unit tests + smoke
gate).  No reward-model exploitation or training instability can be assessed
without extended runs.

## Stage 6 — Enable Controlled End-to-End Behavior Gradients

This stage tests the central thesis claim that control pressure can shape a more
useful reduced world representation.

### Implementation order

1. Unfreeze the ControllerTrunk first, then the temporal model at a lower
   learning rate.
2. Unfreeze spatial pooling and Top-K scoring/selection.
3. Unfreeze tokenizer and encoder last.
4. Every joint update mixes a factual replay batch with imagined trajectories:

   ``L_joint = λ_fact(L_reward_visible + L_reward_masked + βL_KL)
   + λ_critic L_critic + λ_actor L_actor + λ_entropy L_entropy``.

   The factual reward terms are the anchor to real transitions. Critic targets
   based only on imagined returns and Actor gradients are useful task pressure,
   but are not treated as independent grounding signals.
5. Use separate parameter groups and a lower learning rate for newly unfrozen
   shared blocks; this is one continuous guarded schedule, not alternating
   retraining cycles.
6. Measure gradient magnitude and cosine similarity from reward, Actor, and
   Critic losses at each shared block.

### Checkpoint after each unfreeze

- Real return does not regress beyond a predefined tolerance.
- Held-out reward prediction remains acceptable.
- Fixed-probe latent drift remains bounded.
- Attention does not collapse to identical locations.
- Actor/Critic gains persist on unseen CarRacing seeds.
- Held-out visible and masked factual reward probes remain within their
  declared tolerances while imagined return improves.

### Blocker

Re-freeze the latest block if behavior gradients cause predictive forgetting,
latent collapse, exploding gradients, or only imagined improvement.

## Stage 7 — Progressive Collection and Replay

### Implementation

```text
collect real experience with current policy
→ mix new and retained real replay
→ update world model and reward anchor
→ update Actor-Critic in imagination
→ evaluate fixed and unseen seeds
→ repeat
```

- Begin with uniform replay.
- Version every newly collected rollout and record its policy provenance,
  environment seed, terminated/truncated flags, and collector configuration.
- Add priority using a mixture of random coverage, TD error, reward extremes,
  novelty, and rarity only when uniform replay exposes a measured limitation.
- Do not restore exact latent hashes as the main memory lookup.

### Checkpoint

- New-policy data improves coverage without catastrophic loss on old probes.
- Performance improves across cycles on fixed and unseen seeds.
- Resource and environment-interaction budgets are recorded.
- A fixed evaluation protocol reports mean and variance over multiple seeds,
  with distinct fixed-seed, unseen-seed, and optional second-environment
  results.

## Required Progress Metrics

Every stage should record, where applicable:

- train and held-out immediate reward error;
- cumulative reward error by rollout horizon;
- real and imagined return;
- Actor loss, Critic loss, entropy, and value error;
- gradient norm per block and per loss;
- latent drift on fixed probes;
- selected-token diversity and stability;
- parameters, FLOPs or MACs, throughput, latency, and peak GPU memory;
- real-environment interaction count;
- result variance across seeds.
- rollout schema version, dataset manifest ID, configuration ID, and checkpoint
  provenance.

## DreamerV3 Adoption Boundary

Use DreamerV3 as the default source for already-tested stabilization choices:

- imagined Actor-Critic training;
- lambda-return/value targets;
- slow Critic targets;
- return normalization;
- entropy regularization;
- gradient clipping;
- robust reward/value scaling;
- progressive replay and data collection.

Do not copy its full RSSM, decoder, or scale unless a checkpoint demonstrates
that a corresponding mechanism is necessary. Unlike standard DreamerV3, this
project intentionally tests staged Actor-Critic gradients into the temporal and
perceptual model. That difference must remain explicit in experiments.

## Primary References

- Hafner et al., *Mastering diverse control tasks through world models*, Nature
  (2025): https://www.nature.com/articles/s41586-025-08744-2
- Official DreamerV3 implementation and configurations:
  https://github.com/danijar/dreamerv3
