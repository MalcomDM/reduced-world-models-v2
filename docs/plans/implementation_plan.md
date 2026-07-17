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
Six cached runs at K=4/16/32 × seeds 42/43 (beta=0.1, linear reward head,
10 epochs) compared against the K=8 vectorized anchor.

| K   | Seed | Ratio | Mean |
|-----|------|-------|------|
| 4   | 42   | 0.931 | 0.886 |
| 4   | 43   | 0.841 |      |
| 8   | 42   | 0.828 | 0.808 |
| 8   | 43   | 0.787 |      |
| 16  | 42   | 0.834 | 0.822 |
| 16  | 43   | 0.809 |      |
| 32  | 42   | 0.896 | 0.870 |
| 32  | 43   | 0.843 |      |

**Verdict:** K=8 and K=16 are essentially tied (mean ratio 0.808 vs 0.822).
Both clearly outperform K=4 (0.886) and K=32 (0.870).  K=16 costs the same
forward pass (CNN + tokenizer + scorer process all 225 patches regardless of
K), so K=8 remains the default for computational efficiency.  See
``runs/component_refinement/RUN_INDEX.md`` for full details.

### D — Masked dynamics gate

- Warm up on visible observations, mask contiguous horizons 1/2/4/8/16, and
  replay factual action sequences.
- Compare correct, zero, and shifted action histories on controlled branches.
- Decide whether finite Transformer context suffices or generated state must be
  explicitly carried.

### Stage 2.5 checkpoint

- The corrected anchor matches or exceeds the current reward result on the
  fixed suite.
- Retained perception mechanisms beat declared matched baselines across seeds.
- Masked reward predictions degrade gradually with horizon and respond to
  state-conditioned action branches.
- Runtime, MACs, latency, memory, and unique environment frames are reported.

Only then proceed to general open-loop imagination and Actor-Critic training.

## Stage 3 — Implement the Trainable Imagination Engine

Stage 2.5D is the experimental gate for masked dynamics. Stage 3 does not
repeat that validation: it turns the validated observed/masked transition
semantics into a reusable, differentiable imagination interface for later
Actor-Critic training.

### Implementation

- Implement one transition interface for observed and masked/imaged steps,
  using the approved order: score `R(b_t, a_t)`, then advance with `a_t` and
  the available/missing next observation.
- Preserve tensors, gradients, histories, masks, and action timing; no Python
  float conversion or implicit detach inside trainable rollout code.
- Replace the inactive `RolloutSimulator` draft only after it agrees with the
  Stage 2.5D evaluator on fixed factual continuations.
- Define the bounded context/recursive-state policy selected by Stage 2.5D and
  make it explicit in the interface.

### Checkpoint

- Incremental observed steps and the interface's observed mode agree with the
  validated factual model.
- The interface reproduces the fixed masked-horizon evaluator's predictions.
- Rollouts remain tensors with finite gradients and correct action/reward
  phase alignment.
- Deterministic evaluation is reproducible.

### Blocker

Do not train a policy until Stage 2.5D has shown action-sensitive masked
dynamics and this implementation reproduces that result exactly.

## Stage 4 — Implement Actor-Critic with Frozen World Model

### Implementation

- Add the shared Actor branch and Critic branch.
- Define a bounded continuous distribution for steering, gas, and brake.
- Implement stochastic training actions and deterministic evaluation actions.
- Implement log probabilities, entropy, terminal-aware lambda returns/GAE, and
  a slow or target Critic using DreamerV3 patterns where applicable.
- Freeze perception and temporal parameters for this checkpoint.

### Checkpoint

- Distribution samples and deterministic actions respect CarRacing bounds.
- GAE matches hand-calculated trajectories including termination.
- Critic overfits known synthetic returns.
- Positive and negative advantages move action likelihood in the correct
  direction.
- Actor and Critic optimize without changing world-model parameters.

### Blocker

Do not enable upstream behavior gradients until policy and value learning are
correct with stable inputs.

## Stage 5 — Train Actor-Critic in Imagination

### Implementation

- Generate short differentiable imagined trajectories.
- Start with DreamerV3-style return normalization, entropy regularization,
  gradient clipping, and a slow Critic target.
- Make the actor-gradient estimator explicit: dynamics gradients,
  likelihood-ratio gradients, or a documented mixture.
- Train with the world model frozen first.

### Checkpoint

- Predicted return improves across multiple seeds.
- Actor entropy does not collapse immediately.
- Critic error remains bounded as the policy changes.
- Short real-environment evaluations improve with predicted return.

### Blocker

If imagined reward rises while real reward falls, treat this as reward-model
exploitation and return to Stage 2 or shorten the imagination horizon.

## Stage 6 — Enable Controlled End-to-End Behavior Gradients

This stage tests the central thesis claim that control pressure can shape a more
useful reduced world representation.

### Implementation order

1. Unfreeze the temporal model at a lower learning rate.
2. Unfreeze spatial pooling and Top-K scoring/selection.
3. Unfreeze tokenizer and encoder last.
4. Continue direct reward updates from real replay throughout.
5. Measure gradient magnitude and cosine similarity from reward, Actor, and
   Critic losses at each shared block.

### Checkpoint after each unfreeze

- Real return does not regress beyond a predefined tolerance.
- Held-out reward prediction remains acceptable.
- Fixed-probe latent drift remains bounded.
- Attention does not collapse to identical locations.
- Actor/Critic gains persist on unseen CarRacing seeds.

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
