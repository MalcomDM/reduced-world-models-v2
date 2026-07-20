# Factual Memory and Versioned Latent Cache Contract

## Status and scope

This is the canonical design for Stage-7 factual experience memory and its
optional MinimalSRU latent-start cache. It is **not implemented yet**.

The factual memory is central to the wake–dream learning cycle: it controls
which real experiences are retained and replayed. The cached `z_t` layer is
only an efficiency mechanism. The first matched S5 Actor-Critic checkpoint
passed without caching in 49.1 seconds, so cache speed is not itself the
scientific motivation.

The legacy `BehaviorMemory` is not compatible with this contract.

## Record semantics

Each record must contain two separable parts.

### Immutable factual pointer

- dataset-manifest hash and source path/file hash;
- episode identifier/environment seed and timestep `t`;
- timing contract: `z_t` contains `obs[t]`, `action[t-1]`, and earlier state,
  before choosing `action[t]`;
- reconstruction context, including burn-in/visible-context layout;
- behavior-policy provenance;
- immediate reward, terminated/truncated flags, and optional
  horizon-specific factual return `G_t^H`.

The factual pointer survives model updates and is the source of truth.

### Disposable latent cache

- detached CPU `z_t` with explicit shape/dtype;
- cache schema version;
- temporal backend and resolved model-config digest;
- exact world-model parameter hash;
- optional priority inputs such as TD error, novelty, rarity, or reward
  extreme, each with the version that produced it.

For MinimalSRU, `z_t` is the complete recurrent state. No observation or
history buffer is required to resume blind imagination. A legacy causal
Transformer cache would additionally require its bounded history and lengths.

## Construction and validity

- Materialize cached states in evaluation mode with deterministic tokenizer
  mean, `torch.no_grad()`, and `z_t.detach()`.
- A cached state is usable only with the exact backend, config digest, and
  world-model parameter hash that produced it.
- Any observation preprocessing, perception, tokenizer, spatial, or MinimalSRU
  update invalidates the entire `z_t` cache.
- ControllerTrunk and head updates do not invalidate `z_t`, because they are
  downstream of the cached recurrent state. They do invalidate any separately
  cached shared representation `c_t`.
- Actor/Critic/Controller updates may stale value, TD-error, or
  policy-dependent priorities even when `z_t` remains valid.
- Invalidation deletes/replaces latent values, never factual pointers.
  Rebuilding from those pointers must be atomic and reproducible.

## Gradient boundary

- Cached starts may train Actor/Critic cheaply while the world model is frozen.
- No gradient may be claimed through a stored `z_t` into the computation that
  originally produced it.
- Any joint world-model update must load the factual pointer and reconstruct
  the bounded context with the current model inside the active graph.

## Sampling and retention

- Establish an equal-budget uniform replay baseline first.
- If prioritization is justified, keep a nonzero uniform floor and mix
  ordinary, positive, negative, surprising, rare, and terminal strata.
- High reward alone is not a valid replay distribution.
- Use bounded FIFO/reservoir replacement inside each training stratum.
- Keep fixed evaluation probes and rare factual boundary cases outside the
  mutable training-memory population.
- Record mixture weights. If weighting changes the objective, apply/report
  importance correction or state the changed objective explicitly.

## Target interpretation

- A recorded behavior return is evidence and a comparison threshold, not an
  upper bound for the Actor.
- Because the Critic estimates the evolving policy value, behavior-policy
  returns are not automatically unbiased Critic targets.
- Imagined improvements must be checked against factual branch replay and
  locked real-environment evaluation.

## Stage placement

1. S5 proved uncached z-only imagination and Actor-Critic behavior.
2. A later matched efficiency ablation may compare cached versus reconstructed
   starts under identical start samples and imagined-transition budgets.
3. Stage 6 reconstructs factual contexts whenever gradients enter the world
   model.
4. Stage 7 establishes uniform replay, then tests bounded stratified replay as
   the explicit first-cycle policy-bootstrap hypothesis.

## Minimum acceptance tests

- Cached and freshly reconstructed `z_t` agree for the same pointer/hash.
- Wrong hash/backend/config records are rejected, never silently consumed.
- Stored states are detached and contain no retained computation graph.
- Cached and uncached Actor-Critic updates use matched starts and budgets.
- Cache refresh preserves factual pointer identity and provenance.
- Priority refresh is independent from latent invalidation.
