# Stage 7 — Factual Memory and Dream Replay Validation Plan

## Purpose and boundary

Stage 7 tests whether continuous probabilistic selection of informative
factual starts improves Actor-Critic learning efficiency over uniform replay.

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
6. **Use continuous factual priorities first.** Immediate rewards,
   finite-horizon factual returns, termination and reward changes define the
   first priority. TD error/model surprise may be added only with explicit
   model versions.
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
11. **Bound the working set first.** Keep all cheap factual pointers while
    practical; probabilistically rebuild a bounded active dream/cache set each
    cycle.
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
- continuous priority inputs plus non-exclusive descriptive tags.

### Versioned derived metadata

- return change or reward-event score;
- rarity/count statistics and their population digest;
- optional model error, TD error or novelty with producer checkpoint hash;
- sampling weight, probability and last priority-refresh version.

### Optional working cache

- detached CPU `z_t`, dtype and shape;
- MinimalSRU/backend and preprocessing/config digests;
- exact state-producing parameter hash;
- materialization timestamp and cache schema.

## Continuous priority and active-set sampling

For pointer `i`, compute train-only percentile ranks:

```text
qGᵢ = percentile rank of finite-horizon factual return
q⁺ᵢ = max(0, 2qGᵢ - 1)       # upper-half extremeness
q⁻ᵢ = max(0, 1 - 2qGᵢ)       # lower-half extremeness
dᵢ  = mean(r[i:i+h]) - mean(r[i-h:i])
q↑ᵢ = positive-tail percentile of dᵢ, else 0    # recovery/improvement
q↓ᵢ = positive-tail percentile of -dᵢ, else 0   # degradation/failure
```

An initial continuous priority score may use:

```text
sᵢ = λ₊ (ε + q⁺ᵢ)^α
   + λ₋ (ε + q⁻ᵢ)^α
   + λ↑ (ε + q↑ᵢ)^β
   + λ↓ (ε + q↓ᵢ)^β
   + λT terminalᵢ
```

After crowding correction, normalize the score and mix a fixed total uniform
mass:

```text
s'ᵢ = sᵢ / count_same_returnᵢ^ρ
P(i) = η / N + (1 - η) · s'ᵢ / Σⱼs'ⱼ
```

`η=0.1` means ten percent of total probability is uniform; it is not an
additive constant repeated for every pointer. The separated `q⁺`/`q⁻` terms
avoid the degenerate `q + (1-q) = 1` case. Percentiles avoid dependence on
CarRacing's absolute reward scale. The local window `h` uses a short
reward-rate comparison rather than adjacent raw rewards, so ordinary sparse
reward pulses do not all become “surprises.”

`q↑` and `q↓` are priority signals and reporting tags, not reserved memory
capacities. They preserve both recovery and failure transitions while allowing
all records to compete in the same probabilistic active set. Model prediction
error may later add a separate, checkpoint-versioned surprise term; it is not
mixed with this factual change signal.

Normalize `P(i) = wᵢ / Σⱼwⱼ`. Draw the active set without replacement. A
simple reproducible implementation may assign:

```text
keyᵢ = -log(Uᵢ) / wᵢ
```

and retain the `M` smallest keys, where `Uᵢ` is deterministically generated
from `(cycle_seed, pointer_id)`. This is weighted reservoir sampling. `M` is a
compute/cache budget, not a separate quota for reward slices.

The complete factual index remains available. Every cycle generates new keys,
so new high-value pointers can enter and old working memories can leave without
explicit pairwise similarity or destructive deletion. Positive, negative,
ordinary, transition and terminal labels remain useful for reports, not for
hard storage partitions.

This is competition by **factual priority**, not semantic image similarity.
The corpus audit found a dominant exact-return group, so Stage 7 uses a
lightweight equal-return crowding correction:

```text
s'ᵢ = sᵢ / count_same_returnᵢ^ρ
```

The selected initial value is `ρ=0.25`: duplicate priority grows with the
three-quarter power of group size instead of linearly. The corpus audit showed
that `ρ=0.5` over-concentrates probability on rare returns. `ρ={0,0.5,1}`
remain declared ablations. The uniform mixture `η` remains intact. This uses
the already indexed, quantized factual return and is not a learned similarity
model.

## Stage 7.0 — Measurement and uniform foundation

### 7.0A — Corpus inventory

Status: **COMPLETE**.

Build a read-only profiler over eligible training files:

- unique transitions and possible start pointers;
- immediate-reward and finite-horizon-return distributions;
- counts and distributions for smoothed upward/downward reward-rate changes,
  terminal/off-road candidates and rare tags;
- percentile/tie distributions and aggregate priority mass;
- episode/file contribution across priority quantiles.

Deliver a compact report and proposed coefficients, exponents, uniform floor
and active-set size. Do not train.

### 7.0B — Factual pointer index

Status: **COMPLETE**. The post-review integrity pass added immutable public
access, conflicting-ID rejection and explicit source-hash validation.

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

## Stage 7.1 — Probabilistic factual memory

Implement:

- deterministic percentile/rank and equal-return crowding calculations;
- continuous weight calculation and nonzero uniform floor;
- weighted sampling without replacement;
- bounded active-set rebuild with deterministic cycle seeds;
- quantile/tag composition and entry/exit reports;
- priority refresh independent from pointer identity.

Acceptance:

- achieved sampling frequencies match declared probabilities within tolerance;
- no evaluation pointer can be inserted;
- old factual pointers survive model/cache invalidation;
- every sampled item reports probability, weight and diagnostic tags;
- new pointers can enter and old active pointers can leave reproducibly.

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
2. continuous probabilistic-priority replay;
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
- Critic error/calibration by return and priority quantile;
- action entropy, bounds, diversity and state dependence;
- deterministic branch outcomes from selected factual prefixes where replay
  is available;
- visible/masked reward anchors, even though the world model is frozen;
- real returns on the same development seeds;
- straight, curve, off-road and recovery scenario summaries where labels exist.

Primary behavioral success criteria and numerical tolerances are declared
after the corpus inventory but before any policy run.

## Optional future layer — Scenario mastery

The same pointer system can later define a replayable scenario:

```text
scenario_id = (environment_seed, prefix_t, horizon, goal_version)
```

Store attempt count, last attempt, an exponential moving average of achieved
return/success, a target threshold and recent learning progress. A future
continuous need score can combine:

```text
need = λgap · max(0, target - mastery)
     + λprogress · |mastery_now - mastery_old|
     + λstale · staleness
```

Mastered scenarios lose gap priority; stale skills can return; learning
progress favors scenarios the agent can currently improve. Caps and the
uniform floor prevent impossible scenarios from monopolizing replay.

This resembles prioritized level replay and automatic curricula, but it is
**not part of the first Stage-7 implementation**. We only preserve the
scenario/provenance fields needed to add it later. Exact deterministic prefix
replay and generic goal definitions must be validated before activation.

## Stage 7.4 — First wake–dream cycle

After selecting the replay distribution:

1. collect new factual experience with declared exploration;
2. evaluate and insert pointers using the frozen continuous-priority contract;
3. optionally update the world model using factual reward/KL anchors only;
4. invalidate and rebuild `z_t` if perception/SRU changed;
5. consolidate Actor/Critic in dreams;
6. evaluate fixed probes and unseen development tracks.

No Actor/Critic gradient enters SRU or perception in Stage 7.

## Evidence hierarchy

1. **Capacity:** positive-heavy replay can improve selected starts.
2. **Memory contribution:** probabilistic priority beats equal-budget uniform
   replay.
3. **Transfer:** improvement survives held-out factual/scenario probes.
4. **Behavior:** locked development-track return improves.
5. **Cycle:** improvement persists after new experience and cache refresh.

Failure at one level localizes the problem. It does not automatically reject
MinimalSRU, Actor-Critic or the complete architecture.

## Required artifacts

Tracked documentation:

- corpus inventory and frozen priority/sampling protocol;
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
- probabilistic priority replay is compared with uniform under matched budgets;
- positive-heavy capacity result is clearly separated from generalization;
- selected replay improves or meaningfully diagnoses real behavior across two
  model seeds;
- at least one wake–dream refresh preserves factual grounding;
- a stronger frozen behavioral baseline is ready for Stage-8 joint-gradient
  calibration.

## Theoretical precedents and scope

This protocol is a deliberately small synthesis, not a claim of a new
standalone memory algorithm:

- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) motivates
  non-uniform replay while preserving a stochastic sampling floor.
- [Weighted reservoir sampling](https://www.sciencedirect.com/science/article/pii/S002001900500298X)
  provides the reproducible bounded active-set mechanism without fixed reward
  buckets.
- [Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html)
  and [automatic curriculum learning](https://arxiv.org/abs/1704.03003)
  motivate the deferred mastery, learning-progress and staleness fields.

Stage 7 initially uses factual return ranks rather than TD error so its memory
comparison does not depend on an immature Critic. Policy-dependent priorities
remain a later, explicitly versioned extension.
