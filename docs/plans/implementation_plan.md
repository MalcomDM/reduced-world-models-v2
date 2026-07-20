# Reduced World Models — Main Implementation Plan

## Purpose

This is the active engineering and research sequence for the thesis. It records
the selected architecture, current boundary, remaining stages, and advancement
gates. It is intentionally not a diary of every superseded implementation.

Detailed historical evidence remains in:

- [Architecture validation](architecture_validation_plan.md)
- [Completed MinimalSRU decision](sru_temporal_validation_plan.md)
- [Theoretical probe index](../evidence/theoretical_probes.md)
- [Training-budget protocol](../protocols/training_budget_protocol.md)
- [Transition contract](../contracts/transition_contract.md)
- [Latent-memory contract](../contracts/latent_memory_contract.md)

The causal-Transformer Stage-5 baseline is preserved on
`baseline/causal-transformer-stage5`. MinimalSRU is the selected backend on
`main`.

## Selected Architecture

```text
observation o_t
    → CNN encoder
    → stochastic variational patch tokenizer + positional encoding
    → learned attention scorer and differentiable Top-K=8 selection
    → weighted spatial representation p_t

(p_t, previous action a_{t-1}, observation-visible bit, z_{t-1})
    → MinimalSRU
    → recurrent dynamic state z_t

z_t → ControllerTrunk → shared representation c_t
    ├── RewardHead(c_t, current action a_t) → predicted r_{t+1}
    ├── Actor(c_t) → bounded action distribution
    └── Critic(c_t) → expected future return
```

### Why MinimalSRU replaced the causal Transformer

The causal Transformer required a bounded temporal history at every inference
step. That conflicted with the intended observational-dropout behavior and
made a reusable memory start contain an entire history rather than one compact
state.

MinimalSRU makes `z_t` the complete recurrent state:

```text
x_t = concat(observation_keep_t * p_t, a_{t-1}, observation_keep_t)
z_t = SRU(x_t, z_{t-1})
```

Consequences already validated:

- blind evolution continues from `z_t` while observations are absent;
- incremental inference carries only `z_t`;
- an imagined rollout can begin from one version-valid state;
- factual reconstruction remains differentiable when world-model blocks train;
- the temporal block has 5,920 parameters versus 56,560 in the causal baseline;
- frozen Actor-Critic control reached the predefined causal-parity gate.

The selected factual training path uses 20 loss-free burn-in steps followed by
16 supervised targets. Burn-in contributes to the target gradient graph but
does not receive direct reward/KL loss.

## Non-Negotiable Contracts

1. `z_t` contains the current observation and `a_{t-1}` before choosing `a_t`.
2. Reward prediction is `R(z_t, a_t) = r_{t+1}`.
3. Actor and Critic consume the same shared interpretation of `z_t`.
4. Masked observations cannot enter perception or tokenizer KL.
5. Validation files, manual evaluation episodes, and locked test seeds never
   enter training or memory selection.
6. Cached `z_t` is valid only for the exact world-model/config hash that
   produced it.
7. Any claim that behavior losses shape the world model must reconstruct the
   factual context inside the current computation graph.
8. Imagined improvement is not accepted without factual and real-environment
   gates.
9. Every comparison records real transitions, model positions, imagined
   transitions, optimizer updates, wall time, memory, and parameter counts.

## Completed Foundation

| Milestone | Result |
|---|---|
| Stages 0–0.5 | Transition contracts, episode-safe splits, structured configs/checkpoints, deterministic probes. |
| Stages 1–2 | End-to-end factual reward pipeline and held-out baseline improvement. |
| Stage 2.5 | Correct KL reduction, connected Top-K gradients, action timing, performance/cache work, perception and dropout ablations. |
| Stage 3 | Differentiable score-then-advance imagination interface. |
| Stage 4 | Bounded Actor, Critic, TargetCritic, lambda returns and termination semantics. |
| Stage 5 | Frozen imagined Actor-Critic and real-environment evaluator. |
| SRU S0–S6 | MinimalSRU mechanics, burn-in selection, strict observational dropout, z-only imagination and frozen-policy parity. |
| Stage 6.0 | Independent joint-gradient audit with deterministic factual batch, routing, cosine, state restoration and no-update proof. |

Historical counts, intermediate commands, rejected macroblock/cold-start
variants, and causal-line metrics belong in the linked evidence documents, not
in this active plan.

## Stage 6 — Controlled Behavior Gradients

**Status: active. The corrected Stage-6.1 seed-42 experiment is running;
seed-43 replication remains pending.**

Goal: establish one safe shared-representation update boundary before memory
changes the training distribution. Deeper SRU, Actor and perception gradients
are intentionally deferred until Stage 7 produces a stronger behavioral
baseline.

### 6.1 — ControllerTrunk + Critic

Trainable:

- ControllerTrunk and RewardHead at a lower shared learning rate;
- OnlineCritic at its normal learning rate;
- TargetCritic through Polyak updates only.

Frozen:

- Actor;
- MinimalSRU;
- complete perception stack.

Each update mixes visible factual reward MSE, strict masked reward MSE and
Critic loss from short imagined trajectories. Training and validation use
disjoint episode files through the shared split implementation.

**Split-clean two-seed run: COMPLETE; calibration deferred.** Both seeds used split-clean AC prerequisites
(16 train / 4 val files, disjoint). Visible and masked gates pass for both seeds.
Real-return gate passes for seed 43 but fails narrowly for seed 42.

| Gate | Seed 42 | Seed 43 |
|------|:-------:|:-------:|
| Visible delta ≤ +0.02 | PASS (−0.0110) | PASS (−0.0161) |
| Masked delta ≤ +0.02 (all H) | PASS | PASS |
| Real return treatment ≥ control | FAIL (−82.35 vs −80.15) | PASS (identical −82.35) |
| Actor/SRU/perception unchanged | PASS | PASS |

Full report: `runs/imagined_actor_critic/minimal_sru/04_joint_controller_critic_split_clean/RESULTS.md`

All earlier Stage-6.1 runs (`02_joint_controller_critic`, `02_joint_controller_critic_corrected`)
are superseded because their AC prerequisites or data splits were not split-clean.

Gate:

- zero train/validation overlap;
- Actor, SRU and perception bitwise unchanged;
- visible and mean masked ratios regress by at most 0.02;
- real return does not regress against the corresponding frozen control;
- result direction is checked across two seeds.

Protocol: [Stage 6.1 replication](../evidence/joint_training/stage6_1_replication_protocol.md).

### Stage-6 decision

- two-seed evidence for the ControllerTrunk + Critic predictive boundary
  (visible and masked gates pass both seeds);
- real-return gate passes seed 43 but fails seed 42 by 2.2 points;
- the proposed joint weighting/LR is mechanically stable but too coarse to
  adopt unchanged;
- Actor, MinimalSRU and perception remain intentionally frozen;
- no validation leakage or stale-latent gradient claims.

Stage 6 is experimentally complete as a connectivity and calibration probe.
ControllerTrunk + Critic joint updates preserve prediction, but the particular
500-update schedule did not satisfy real-return non-inferiority in both model
seeds. This does **not** reject joint learning: updating one head aggressively
while the others remain fixed can temporarily shift their shared
representation. Stage 7 begins from the split-clean frozen controls to avoid
that confound. Stage 8 will retest the same boundary using fewer updates
between validations, smaller LR/critic weight, or interleaved factual and
behavior updates, with early stopping on the predictive and real gates.

Critic pressure into MinimalSRU, Actor pressure into shared representations,
and progressive perception unfreezing move to Stage 8. They must be evaluated
against the stronger memory-trained policy rather than the current weak
frozen-policy baseline.

## Stage 7 — Experience Memory and Replay

Memory is a central Stage-7 component, not a late optional convenience.
Stage 6 validated the ControllerTrunk + Critic machinery but deferred schedule
calibration. Stage 7 builds a stronger behavioral baseline with the world model
and ControllerTrunk frozen during the matched memory-selection experiment.
After selection, the first wake–dream refresh may apply factual reward/KL
updates, but no behavior gradient enters upstream blocks. This makes later
joint-gradient regressions materially measurable.

Detailed protocol:
[Stage-7 memory and replay plan](memory_replay_validation_plan.md).

### 7.0 — Uniform Collection/Replay Baseline

```text
collect with current policy
→ combine new and retained factual replay
→ update world model under factual anchors
→ consolidate Actor/Critic in dreams
→ evaluate fixed and unseen seeds
```

Establish one equal-budget uniform replay cycle before claiming that smart
selection helps. Persist policy/model provenance, environment seed,
termination, source hashes and transition budgets.

### 7.1 — Probabilistic Factual Memory

The permanent unit is a factual pointer, not an activation:

- episode/file hash and timestep;
- required reconstruction context;
- policy and world-model versions;
- immediate reward and horizon-specific future returns;
- termination/truncation;
- optional TD error, novelty, rarity and return-change metadata.

Use continuous train-only percentile/rank signals:

- high and low finite-horizon factual return;
- directional, locally smoothed reward-rate changes (recovery and degradation);
- termination or rare-boundary bonuses;
- later, versioned learning-progress/mastery signals.

Preserve a nonzero uniform floor and rebuild a bounded active dream set through
weighted sampling without replacement. The complete factual-pointer index
remains cheap and permanent; old active memories leave naturally when new
weighted samples are drawn. Reporting tags may describe positive, negative,
ordinary, change and terminal cases, but they do not impose fixed slice
budgets. A mild equal-return crowding correction reduces redundant priority
without suppressing the uniform floor. Positive-only replay is
permitted only as an explicitly labelled overfit diagnostic.

### 7.2 — Versioned `z_t` Working Cache

While the world model is frozen, materialize deterministic, detached CPU `z_t`
for selected factual pointers. This removes repeated perception and 20-step
burn-in during large dream budgets.

Any state-producing world-model update invalidates the entire cache. Rebuild
from factual pointers atomically. Actor/Critic-only changes do not invalidate
`z_t`, although policy/value-dependent priorities may need refreshing.

Gate:

- cached and reconstructed starts agree for the same pointer/hash;
- wrong hashes/configs/backends are rejected;
- equal-start/equal-imagined-transition cached and uncached training agree;
- measured speedup justifies making caching the default.

### 7.3 — Memory-Selection Experiment

Compare equal budgets:

1. uniform factual starts;
2. continuous probabilistic-priority memory;
3. positive-heavy overfit diagnostic.

Measure:

- Critic fit and calibration by return/priority quantile;
- Actor imagined return and action diversity;
- deterministic factual branch outcomes from stored starts;
- locked real-environment return;
- curve, straight-road, off-road and recovery behavior;
- representation drift and forgetting.

This directly tests whether rare useful random-policy windows can bootstrap a
better first policy.

### 7.4 — Wake–Dream–Integrate Cycle

1. **Wake:** collect factual experience with the current Actor plus declared
   exploration.
2. **Ground:** optionally update reward/dynamics representations using factual
   reward/KL anchors only.
3. **Materialize:** rebuild continuous pointer priorities and the versioned
   active cache.
4. **Dream:** cheaply consolidate Actor/Critic from cached starts.
5. **Consolidate:** train Actor/Critic from selected starts; keep behavior
   gradients out of ControllerTrunk, MinimalSRU and perception.
6. **Evaluate:** fixed probes, unseen seeds, resource budget and regressions.

A later optional scenario curriculum may index deterministic
`(environment seed, prefix, horizon, goal version)` records and prioritize
unmastered/stale scenarios. Stage 7 only preserves the metadata needed for
that extension; it does not block the initial memory experiment.

### Stage-7 exit

- probabilistic replay beats the equal-budget uniform baseline;
- at least two collection cycles improve real behavior without forgetting;
- memory replacement and invalidation are deterministic and tested;
- behavior improvements survive unseen tracks/seeds.

## Stage 8 — Memory-Informed Upstream Gradients and Cycle Calibration

Goal: use the stronger Stage-7 policy/memory baseline to decide how far Actor
and Critic may shape the world model, then turn the selected boundary into one
repeatable learning engine.

### 8.0 — Recalibrate ControllerTrunk + Critic Pressure

Repeat the validated Stage-6 boundary against the memory-trained control using
shorter update bursts and validation between bursts. Compare smaller
Controller LR, lower Critic loss weight, and interleaved factual anchoring
without changing several factors in one run.

Select the smallest effective pressure that preserves visible, masked and real
behavior gates. The Stage-6 result is the high-dose reference, not evidence
that the boundary is invalid.

### 8.1 — Critic Pressure into MinimalSRU

Open MinimalSRU at a lower LR while perception and Actor remain frozen.
Compare against the Stage-7 frozen-SRU control using identical memory starts,
factual transitions, imagined transitions and real interactions.

Additional gates:

- recurrent state/action sensitivity does not regress;
- visible and blind factual reward gates remain within tolerance;
- latent/output drift on fixed factual pointers is bounded;
- real improvement is not explained only by reward-head drift.

### 8.2 — Actor Pressure into Shared Representations

Enable Actor and entropy gradients into ControllerTrunk, then MinimalSRU in
separate experiments. Do not open both boundaries in the first run.

Gate:

- Actor remains state-dependent and does not collapse to a constant action;
- entropy and action-bound rates remain healthy;
- Critic calibration remains stable;
- real return improves or stays within tolerance;
- factual reward and masked dynamics remain grounded.

### 8.3 — Progressive Perception Unfreezing

Only if 8.0–8.2 pass:

1. spatial pooling;
2. attention scorer/Top-K surrogate;
3. tokenizer;
4. CNN encoder.

Open one boundary at a time. Reward/KL anchors remain active. Re-freeze the
latest block if it produces predictive forgetting, attention collapse,
exploding gradients, or imagined-only improvement.

### 8.4 — Full-Cycle Calibration

- Define the ratio of factual world-model, cached dream, and reconstructed
  joint updates per cycle.
- Calibrate exploration, memory refresh, TargetCritic update and learning-rate
  schedules.
- Test whether new experience repairs reward-model exploitation.
- Scale horizons only after short-horizon gates remain stable.
- Run multiple cycles and multiple seeds with fixed interaction budgets.

Stage-8 exit:

- monotonic or statistically credible behavior improvement across cycles;
- no progressive reward/masked-dynamics collapse;
- update budgets and stopping rules are fixed before final evaluation.

## Stage 9 — Thesis Ablations and Interpretation

Run matched ablations only on the selected full-cycle configuration:

- learned versus fixed/random Top-K;
- K=4/8/16 where computationally meaningful;
- stochastic tokenizer/KL variants;
- observational dropout on/off and horizon variation;
- frozen versus Critic-shaped versus Actor+Critic-shaped representations;
- uniform versus probabilistic-priority memory versus cached starts;
- selected joint-gradient boundary versus the frozen-world-model control.

Interpretability outputs:

- real-time attention/selected-patch overlays;
- inference-time K variation;
- blind-horizon behavior;
- latent/output drift on fixed factual pointers;
- optional frozen-state decoder as qualitative information-retention evidence.

Stage-9 exit: every retained architectural contribution has matched evidence
or is explicitly classified as an unproven design choice.

## Stage 10 — Final Thesis Evaluation

- Multiple training seeds and unseen CarRacing tracks.
- Fixed real-interaction and imagined-transition budgets.
- Mean, variance and confidence intervals for real return.
- Reward, value and cumulative-return calibration by scenario.
- Parameters, MACs/FLOPs, throughput, latency and peak memory.
- Comparison with the causal baseline, original World Models and appropriately
  scaled DreamerV3 references.
- Reproducible configs, manifests, checkpoints, plots and failure cases.
- Explicit limitations: single environment, reward sparsity, action
  sensitivity, memory bias and any remaining policy collapse.

The “solved” threshold and all final comparison budgets must be declared before
these runs begin.

## First-Cycle Driving Hypothesis

### Stretch hypothesis

A small number of useful transitions in the initial random-policy corpus,
combined with probabilistic-priority memory and cheap dream consolidation, may
be enough
for the first learned policy to acquire basic forward motion and partial road
following.

This is plausible because:

- the reward model already distinguishes sustained reward-producing behavior
  in manual traces after minutes of training;
- strict observational dropout produces usable blind recurrent states;
- z-only dream training is fast;
- the Actor-Critic and joint-gradient routes are mechanically validated;
- policy pressure can now be tested without replacing factual anchors.

### Why “decent driving after one cycle” is not guaranteed

No current result proves that the initial corpus contains enough
**action-conditioned coverage** to infer steering corrections. Existing
correct-versus-shifted action differences are small. A reward model can identify
“on road and moving” without learning how alternative actions recover from a
specific curve.

Additional risks:

- positive random windows may show outcomes without useful counterfactual
  actions;
- reward magnitude/calibration remains imperfect;
- positive-heavy replay can teach a constant-action shortcut;
- Critic targets from an old behavior policy are off-policy for the evolving
  Actor;
- dream optimization can exploit reward-model errors faster than new factual
  data corrects them;
- curves, braking and recovery may be absent or severely underrepresented.

Therefore:

- **basic forward/road preference after cycle one is a realistic research
  target;**
- **consistent cornering or “decent driving” is an ambitious stretch target;**
- failure would not immediately reject the architecture—it may identify a
  coverage, counterfactual-action, calibration or exploration bottleneck.

The first memory experiment must measure the hypothesis rather than assume it:
uniform versus probabilistic-priority starts, positive-heavy overfit as a
diagnostic, and locked factual/real evaluation after equal budgets.

## Required Metrics

Every applicable stage records:

- visible and masked held-out reward MSE/MAE/baseline ratio;
- real and imagined return by horizon and seed;
- Critic error, Actor loss, entropy and action-bound rates;
- gradient norm/cosine by loss and shared block;
- latent/output drift on fixed factual pointers;
- memory composition, sampling probability and replacement counts;
- unique real transitions, model positions, imagined transitions and updates;
- wall time, throughput, inference latency, parameters and peak memory;
- dataset/config/checkpoint hashes and policy provenance.

## DreamerV3 Adoption Boundary

Adopt established stabilization ideas when a measured need appears:
lambda returns, slow value targets, entropy regularization, gradient clipping,
robust scaling and progressive replay.

Do not copy DreamerV3’s full RSSM, decoder or training scale by default. This
thesis intentionally tests reduced perception, emergent recurrent dynamics,
memory-selected dreams and gated behavior gradients into shared
representations.

## Primary References

- Hafner et al., *Mastering diverse control tasks through world models*,
  Nature (2025).
- Official DreamerV3 implementation.
- Ha and Schmidhuber, *World Models*.
- Ha, *Recurrent World Models Facilitate Policy Evolution* and related
  observational-dropout work referenced in the thesis evidence.
