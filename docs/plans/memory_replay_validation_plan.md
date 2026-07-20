# Stage 7 — Factual Memory and Dream Replay Validation Plan

## Purpose and boundary

Stage 7 tests whether selecting informative factual starts improves
Actor-Critic learning efficiency over uniform replay.

It begins from the split-clean frozen controls:

- strict-observational-dropout MinimalSRU anchors for seeds 42 and 43;
- corresponding split-clean frozen Actor-Critic checkpoints;
- only each checkpoint's 16 training files may populate training memory;
- validation files and evaluation episodes remain outside memory.

During the initial memory-selection comparison, perception, MinimalSRU,
ControllerTrunk and RewardHead are frozen. Only Actor, OnlineCritic and
TargetCritic evolve. This isolates memory selection from joint world-model
training. Stage 8 later recalibrates upstream gradients using the stronger
behavioral baseline produced here.

The canonical lifecycle and cache semantics remain in
[the memory contract](../contracts/latent_memory_contract.md).

## Golden rules

1. **Facts are permanent; activations are disposable.** A memory record is
   anchored by an episode/file hash and timestep. Cached `z_t` is never the
   source of truth.
2. **No evaluation leakage.** Validation files, manual benchmark episodes and
   locked test seeds cannot enter memory, priority statistics or normalization.
3. **Uniform control first.** Smart selection is compared against an
   equal-budget uniform sampler from the same eligible pointers.
4. **Matched budgets.** Conditions use the same initial checkpoints, unique
   factual pool, sampled starts, imagined transitions, optimizer updates,
   horizons and real-environment interactions.
5. **Keep a uniform floor.** The default memory cannot become positive-only or
   discard ordinary state coverage.
6. **Strata use factual evidence first.** Immediate rewards, finite-horizon
   factual returns, termination and reward changes define the first index.
   TD error/model surprise may be added only with explicit model versions.
7. **Behavior return is evidence, not an Actor target.** It is neither the
   optimal return nor automatically an unbiased Critic target for a new policy.
8. **Cached `z_t` has no historical gradient.** It may train Actor/Critic and
   downstream Controller blocks, but claims about perception/SRU gradients
   require reconstruction from the factual pointer.
9. **Exact cache versioning.** Backend, preprocessing/config digest and
   state-producing model hash must match. Mismatch is an error, never fallback.
10. **Dependency-aware invalidation.** Perception or SRU changes invalidate
    `z_t`; Actor/Critic/Controller changes only refresh their dependent
    priorities. A cached `c_t` would also depend on ControllerTrunk.
11. **Bounded retention.** Use deterministic reservoir/FIFO behavior within
    each declared stratum and record every replacement.
12. **Overfitting is a diagnostic.** Positive-heavy replay may test capacity,
    but cannot become the selected training distribution without matched
    generalization and real-environment evidence.
13. **No long run before review.** Each implementation checkpoint stops after
    unit tests, schema validation and smoke execution.

## Memory record

### Immutable factual pointer

- schema version and stable record ID;
- dataset-manifest, source-file and episode hashes;
- source timestep and required 20-step reconstruction context;
- environment seed and behavior-policy provenance when available;
- action/reward timing-contract version;
- immediate reward, terminated and truncated separately;
- declared horizon-specific factual returns;
- primary stratum plus non-exclusive descriptive tags.

### Versioned derived metadata

- return change or reward-event score;
- rarity/count statistics and their population digest;
- optional model error, TD error or novelty with producer checkpoint hash;
- sampling weight and last priority-refresh version.

### Optional working cache

- detached CPU `z_t`, dtype and shape;
- MinimalSRU/backend and preprocessing/config digests;
- exact state-producing parameter hash;
- materialization timestamp and cache schema.

## Initial strata

The data-audit checkpoint determines thresholds and capacities before policy
training. It must support at least:

- ordinary coverage;
- positive/high factual return;
- negative/off-road/terminal behavior;
- rapid reward/return changes;
- rare or boundary cases.

Records may have multiple tags but exactly one deterministic primary stratum
for capacity and replacement accounting. Thresholds, precedence, mixture
weights and uniform-floor probability are frozen before the matched runs.

No fixed memory size is assumed in advance. The inventory first reports how
many independent pointers actually exist in every stratum; capacity is then
chosen so minority strata are meaningful without duplicating a handful of
events excessively.

## Stage 7.0 — Measurement and uniform foundation

### 7.0A — Corpus inventory

Build a read-only profiler over eligible training files:

- unique transitions and possible start pointers;
- immediate-reward and finite-horizon-return distributions;
- counts for reward changes, terminal/off-road candidates and rare tags;
- overlap among candidate strata;
- episode/file contribution to each stratum.

Deliver a compact report and proposed thresholds. Do not train.

### 7.0B — Factual pointer index

Implement the versioned pointer schema, deterministic index builder and
uniform sampler.

Acceptance:

- stable IDs and byte-identical rebuild from the same manifest;
- train-only source enforcement;
- transition/timing and episode-boundary correctness;
- deterministic sampling by seed;
- no duplicate pointer unless sampling with replacement is declared;
- serialization round trip and source-hash rejection.

### 7.0C — Uniform dream baseline

Starting from each split-clean frozen control, train Actor-Critic from uniformly
sampled factual starts. This is the control for every smart-memory claim.

Persist exact pointer IDs sampled per update so another sampler can reproduce
the same budget, even when its distribution differs.

## Stage 7.1 — Stratified factual memory

Implement:

- deterministic primary-stratum assignment;
- bounded per-stratum capacities;
- FIFO/reservoir replacement;
- configurable mixture weights and nonzero uniform floor;
- composition and replacement reports;
- priority refresh independent from pointer identity.

Acceptance:

- achieved sampling frequencies match declared probabilities within tolerance;
- empty/minority strata have explicit fallback behavior;
- no evaluation pointer can be inserted;
- old factual pointers survive model/cache invalidation;
- every sampled item reports probability and stratum.

## Stage 7.2 — Versioned `z_t` working cache

Materialize deterministic tokenizer-mean MinimalSRU states for indexed starts.

Acceptance:

- cached and freshly reconstructed `z_t` agree for the same pointer/hash;
- tensors are detached CPU values with no retained graph;
- wrong backend/config/model hash is rejected;
- perception/SRU update invalidates cache atomically;
- Actor/Critic/Controller-only update preserves `z_t` validity;
- cached versus reconstructed Actor-Critic steps agree under matched RNG;
- throughput and memory are measured before cache becomes the default.

The cache is useful only if measured. Memory selection remains scientifically
valid without it.

## Stage 7.3 — Matched memory-selection experiment

Compare from identical initial checkpoints:

1. uniform factual starts;
2. stratified mixture;
3. positive-heavy capacity/overfit diagnostic.

The third condition is not a candidate default. It answers whether the Actor
and Critic can exploit known useful starts at all.

Matched quantities:

- eligible factual pool and split;
- number of sampled starts;
- imagined transitions by horizon;
- Actor/Critic optimizer updates;
- entropy coefficient and TargetCritic schedule;
- evaluation episodes and real interaction count;
- wall-clock and peak-memory reporting.

Measurements:

- imagined return and reward-model exploitation gap;
- Critic error/calibration by primary stratum;
- action entropy, bounds, diversity and state dependence;
- deterministic branch outcomes from selected factual prefixes where replay
  is available;
- visible/masked reward anchors, even though the world model is frozen;
- real returns on the same development seeds;
- straight, curve, off-road and recovery scenario summaries where labels exist.

Primary behavioral success criteria and numerical tolerances are declared
after the corpus inventory but before any policy run.

## Stage 7.4 — First wake–dream cycle

After selecting the replay distribution:

1. collect new factual experience with declared exploration;
2. evaluate and insert pointers using the frozen stratum contract;
3. optionally update the world model using factual reward/KL anchors only;
4. invalidate and rebuild `z_t` if perception/SRU changed;
5. consolidate Actor/Critic in dreams;
6. evaluate fixed probes and unseen development tracks.

No Actor/Critic gradient enters SRU or perception in Stage 7.

## Evidence hierarchy

1. **Capacity:** positive-heavy replay can improve selected starts.
2. **Memory contribution:** stratified beats equal-budget uniform replay.
3. **Transfer:** improvement survives held-out factual/scenario probes.
4. **Behavior:** locked development-track return improves.
5. **Cycle:** improvement persists after new experience and cache refresh.

Failure at one level localizes the problem. It does not automatically reject
MinimalSRU, Actor-Critic or the complete architecture.

## Required artifacts

Tracked documentation:

- corpus inventory and frozen threshold/mixture protocol;
- implementation report and test results;
- matched comparison table and limitations;
- concise thesis-facing theoretical-probe entry.

Ignored run artifacts:

- memory index/cache binaries;
- sampled-pointer logs;
- checkpoints and metrics;
- per-seed evaluation JSON/CSV and plots.

Every report records source manifests, checkpoint/config hashes, seeds, unique
real transitions, starts, imagined transitions, optimizer updates, wall time
and peak memory.

## Stage-7 exit

- memory mechanics and invalidation tests pass;
- stratified replay is compared with uniform under matched budgets;
- positive-heavy capacity result is clearly separated from generalization;
- selected replay improves or meaningfully diagnoses real behavior across two
  model seeds;
- at least one wake–dream refresh preserves factual grounding;
- a stronger frozen behavioral baseline is ready for Stage-8 joint-gradient
  calibration.
