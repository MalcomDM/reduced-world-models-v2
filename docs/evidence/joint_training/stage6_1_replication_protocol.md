# Stage 6.1 — ControllerTrunk + Critic Joint Training: Replication Protocol

## Purpose

The first deliberately narrow end-to-end step. Factual reward losses and
imagined Critic pressure may update the ControllerTrunk, while perception,
MinimalSRU, Actor, and TargetCritic remain frozen.

## Seeds

| Seed | Purpose |
|------|---------|
| `data_split_seed` | Deterministic file-level train/validation partition. Changing this changes the split. |
| `training_seed` | Initialization, DataLoader order, stochastic operations. Changing this does not change the split. |

Both default to 42. They are persisted separately in `training_summary.json`.

## Data

- All eligible `.npz` files under `data/rollouts/rwm_deterministic/scenario_0`.
- Deterministic shuffle at `data_split_seed`.
- Validation ratio from checkpoint config (default 0.2).
- **Training loader uses only train files.** Validation files never enter the training DataLoader.
- Shared split logic: `rwm.data.split.collect_and_split`.

## Frozen Invariants

| Component | Trainable | Params |
|-----------|:---------:|:------:|
| ControllerTrunk | **Yes** | ≈0.6k (encode) |
| RewardHead | **Yes** (via ControllerTrunk) | ≈1.2k |
| Online Critic | **Yes** | ≈5.2k |
| Target Critic | No | Polyak from online |
| Actor | No | Frozen |
| MinimalSRU | No | Frozen |
| SpatialAttentionHead | No | Frozen |
| AttentionScorer | No | Frozen |
| Top-K Selector | No (no params) | — |
| Tokenizer | No | Frozen |
| Encoder | No | Frozen |

## Predeclared Experiments

### Seed 42 (superseded rerun)

Status: **SUPERSEDED — INDIRECT AC SPLIT LEAKAGE**
Path: `runs/imagined_actor_critic/minimal_sru/02_joint_controller_critic_corrected/seed42/`

The Stage-6.1 DataLoader itself was disjoint, but its Stage-5 Actor-Critic
prerequisite had been trained from starts drawn from all 20 rollout files.
Because its Critic supplies gradients to ControllerTrunk, the held-out claim
is not fully isolated. The numerical result remains exploratory and must be
reproduced from a split-clean Actor-Critic prerequisite.

### Split-clean replication (seeds 42 & 43)

Both seeds now have split-clean AC prerequisites (16 train / 4 val files,
disjoint). The joint training was run at:

- Seed 42: `runs/imagined_actor_critic/minimal_sru/04_joint_controller_critic_split_clean/seed42/`
- Seed 43: `runs/imagined_actor_critic/minimal_sru/04_joint_controller_critic_split_clean/seed43/`

Results:

| Gate | Seed 42 | Seed 43 |
|------|:-------:|:-------:|
| Visible delta ≤ +0.02 | PASS (−0.0110) | PASS (−0.0161) |
| Masked delta ≤ +0.02 (all H) | PASS | PASS |
| Real return treatment ≥ control | FAIL (−82.35 vs −80.15) | PASS (identical −82.35) |
| Actor/SRU/perception unchanged | PASS | PASS |
| Train/val disjoint | PASS | PASS |

For seed 42 the control mean is pulled up by seed 101 (−76.90 vs the treatment's
−83.50), causing the real gate failure. Seeds 100 and 102 are identical between
control and treatment for both model seeds.

The threshold remains unchanged. Stage 7 starts from the split-clean frozen
control checkpoints to isolate the memory intervention. The gradient boundary
remains viable; its update magnitude/cadence is recalibrated in Stage 8 after
memory establishes a stronger behavioral baseline.

All earlier Stage-6.1 runs (`02_joint_controller_critic`, `02_joint_controller_critic_corrected`)
are superseded by these split-clean results.

## Advancement Criteria

| Gate | Criterion |
|------|-----------|
| Visible ratio | `treatment − control ≤ +0.02` |
| Correct-action masked ratio | `treatment − control ≤ +0.02` at every horizon |
| Real return (seeds 100,101,102) | Must not regress from corresponding frozen control |
| Losses/gradients | All finite |
| Train/validation overlap | None (verified by `training_summary.json`) |

Negative prediction-ratio deltas are improvements. The `+0.02` threshold is a
non-inferiority ceiling, not a required negative change.

## Pre/Post Evaluation

Using the protocol evaluator (`scripts/evaluation/evaluate_checkpoint.py`) and
masked factual evaluator (`scripts/evaluation/evaluate_masked_dynamics.py`) on
the held-out validation files (untouched by training):

- Visible reward ratio (canonical mask)
- Correct/zero/shifted-action masked ratios at the declared horizons
- Mean reward prediction MSE/MAE

The training CLI additionally rejects an Actor-Critic checkpoint whose
recorded anchor hash does not exactly match the selected world-model anchor.
It verifies that ControllerTrunk, OnlineCritic and TargetCritic actually
update, while Actor, MinimalSRU and perception remain bitwise unchanged.

Real environment evaluation (`scripts/evaluation/evaluate_real_env.py`) on
locked dev seeds 100, 101, 102:
- Actor mode
- Zero-action baseline
- Deterministic random-action baseline

## Blocker

If the seed-43 frozen AC checkpoint does not exist, Stage 6.1 seed 43 is
blocked until the prerequisite SRU anchor and S5 frozen AC are produced.
