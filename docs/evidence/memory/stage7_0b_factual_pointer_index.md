# Stage 7.0B — Versioned Factual Pointer Index

## Summary

Versioned factual-pointer archive with streaming-ingestion API, priority
finalization, deterministic uniform active-set control and atomic
persistence.  Core formulas extracted into `priority.py` with exact
numerical parity to the Stage 7.0A profiler.

## Files changed

| File | Change |
|------|--------|
| `src/rwm/memory/priority.py` | **New** — shared formula primitives extracted from profiler |
| `src/rwm/memory/schema.py` | **New** — `FactualPointer`, `ArchiveEntry`, `PriorityConfig` |
| `src/rwm/memory/index.py` | **New** — `FactualArchive`, `EpisodeIngester`, `build_from_npz`, `UniformSampler` |
| `src/rwm/memory/corpus_profiler.py` | Refactored to import primitives from `priority.py`; full parity |
| `tests/unit/test_memory_index.py` | **New** — behavioral coverage for identity, serialization, isolation, immutable access, conflict rejection, source validation, probabilities, sampling and parity |
| `runs/component_refinement/memory/01_factual_pointer_index/` | Local artifacts (not tracked) |
| `docs/evidence/memory/stage7_0b_factual_pointer_index.md` | This document |

## Schema and record-ID formula

### FactualPointer (immutable)

```
schema_version: int = 1
record_id: str                    # f"{source_hash}:{timestep}"
dataset_manifest: str             # collect_and_split(seed=N)
data_split_seed: int
source_path: str                  # data-root-relative
source_hash: str                  # SHA-256 of NPZ content
episode_id: str                   # file stem
timestep: int
source_episode_length: int
timing_contract_version: str = "1.0"
behavior_policy: Optional[str]
immediate_reward: float
legacy_done: bool
terminated: Optional[bool] = None
truncated: Optional[bool] = None
factual_return_H12: Optional[float]   # quantized to 8 decimals
directional_change_h4: Optional[float]
```

### Stable record identity

```
record_id = f"{source_hash}:{timestep}"
```

Uses only the content-addressed file hash and in-episode timestep.
Project-directory relocation does not change record IDs.  Identical
episode content produces identical IDs.

## Ingestion/finalization lifecycle

```
1. EpisodeIngester(source_path, source_hash, ...)
2. for each transition:          ingester.add_transition(reward, done)
3. all transitions ingested:     pointers = ingester.finalize(H=12, h=4)
4. archive.add_pointers(pointers)    # idempotent
5. archive.finalize()                # recomputes all priorities
6. archive.save(path)                # atomic JSON via os.replace()
```

The ingester buffers only `reward` and `done` arrays — never observations.
Metrics (returns, directional changes) are computed at finalization.

## Priority formula and parity evidence

### Canonical configuration

```
H = 12, h = 4
eta = 0.1         # total uniform probability mass
lambda_pos = lambda_neg = lambda_up = lambda_down = 1.0
lambda_legacy_done = 0
alpha = 1.0, beta = 1.0
rho = 0.25        # equal-return crowding
active_set_M = 1024
quantize_decimals = 8
```

### Formula

```
score_i = lambda_pos * q_pos_i + lambda_neg * q_neg_i
        + lambda_up * q_up_i + lambda_down * q_down_i

crowded_score_i = score_i / count_same_quantized_return_i^rho  (rho=0.25)

P(i) = eta/N + (1-eta) * crowded_score_i / sum(crowded_score)
```

When all scores are zero, falls back to exact uniform `1/N`.

### Parity evidence

`test_probabilities_match_profiler` in `test_memory_index.py` verifies that
the archive's ESS for rho=0 matches the profiler's ESS for the same corpus
and config within 1%.  All formula primitives (`percentile_rank`,
`compute_priority_score`, `compute_probabilities`, `quantize`, etc.) are
shared via `priority.py`.

## Uniform sampler contract

```
key_i = -log(U_i)              # unit weights
U_i   = SHA-256(cycle_seed, record_id) mapped to (0,1)
keep M smallest keys
```

- Deterministic: same seed + same archive → identical sample
- Stable on insertion: existing pointers' keys don't change (verified by test)
- No duplicates
- M=1024 default, configurable
- Sampler provenance recorded in `UniformSampler.last_sample`

## Test results

```
$ pytest -q tests/unit/test_memory_index.py tests/unit/test_corpus_profiler.py
107 passed

$ pytest -q tests/unit
777 passed

$ pytest tests/unit/test_corpus_profiler.py
68 passed in 2.90s

$ pytest tests/unit
767 passed in 68.33s   (no regressions)
```

| Test area | Count |
|-----------|:-----:|
| Stable record IDs | 3 |
| Byte-identical serialization | 3 |
| Train-only isolation | 2 |
| Hand-calculated H=12 / h=4 | 3 |
| Episode boundaries | 2 |
| Idempotent insertion, conflict rejection and immutable access | covered |
| Missing-source and source-hash validation | covered |
| No observation decoding | 1 |
| Probabilities sum=1, positivity | 2 |
| eta uniform mass | 1 |
| rho crowding | 1 |
| Uniform fallback | 1 |
| Priority refresh | 1 |
| Uniform sampler | 3 |
| Atomic persistence | 1 |
| Legacy done fields | 2 |
| Numerical parity with profiler | 1 |

## Seed-42/43 smoke results

| Metric | Seed 42 | Seed 43 |
|--------|:-------:|:-------:|
| Pointers | 4280 | 4546 |
| Build time | 0.16s | 0.16s |
| Finalize time | 0.05s | 0.06s |
| Archive JSON size | 4.4 MB | 4.7 MB |
| Deterministic digest | matches | matches |
| Prob sum | 1.000 | 1.000 |
| Prob min | 2.34e-5 | 2.20e-5 |
| Prob max | 2.80e-3 | 2.70e-3 |
| ESS | 1961 | 2048 |
| Missing H=12 | 176 | 176 |
| Missing h=4 | 112 | 112 |
| Legacy done | 0 | 0 |
| Train/val disjoint | yes | yes |
| Observations decoded | no | no |

Missing H=12 and h=4 metrics correspond to pointers near episode starts
(timesteps < h or insufficient future steps for H=12).

## Uniform M=1024 sample digest

Deterministic across rebuilds:

- Seed 42: `fe9e68ed118df190...`
- Seed 43: `94dc67259b82bf64...`

## Artifact paths

```
runs/component_refinement/memory/
├── 00_corpus_inventory/          ← Stage 7.0A (local, not tracked)
├── 01_factual_pointer_index/     ← Stage 7.0B (local, not tracked)
│   ├── seed42/archive.json
│   ├── seed42/corpus_summary.json (if regenerated)
│   ├── seed43/archive.json
│   ├── seed43/corpus_summary.json
│   └── RESULTS.md
└── RUN_INDEX.md
```

## Remaining blockers or decisions

1. **Terminated/truncated unavailable in the legacy corpus**: The current
   rollout files store only `done = terminated OR truncated`, so both fields
   remain `None` for those files. The streaming ingestion API already preserves
   separate values when future collectors provide them; legacy data is never
   relabelled by inference.

2. **Observational-dropout anchor hash**: The archive records source-file
   hashes but does not yet record the world-model parameter hash or
   config digest needed for `z_t` cache validation. This belongs in
   Stage 7.2 (latent cache), not here.

3. **Priority refresh is intentionally global at this scale**:
   `finalize()` recomputes all percentile-relative metadata after insertion.
   This is exact and costs about 0.2 seconds for the current 4–5k-pointer
   corpus. An approximate incremental rank structure is deferred until scale
   measurements justify its added complexity.

## Integrity correction after review

- conflicting reuse of a stable record ID is rejected;
- archive accessors and serialized dictionaries cannot mutate stored pointers;
- pointer schema, ID, split seed, relative path and timestep bounds are checked;
- persisted priority configuration rejects invalid ranges;
- `validate_sources()` rejects missing or hash-mismatched rollout files;
- atomic save flushes and fsyncs before replacement;
- NPZ files are closed deterministically after reading reward/done arrays.
