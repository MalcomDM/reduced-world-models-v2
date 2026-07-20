# Stage 7.0A — Factual Memory Corpus Inventory (Corrected)

## Summary

Read-only profiler over eligible training-file transitions for Stage 7
memory. This is the **corrected** version addressing: single-horizon qG,
quantized tie convention, corrected episode-end semantics, neutral reward
description, proper density/concentration metrics, real active-set
simulation, bounded M, and Git-artifact hygiene.

## Files changed

| File | Change |
|------|--------|
| `src/rwm/memory/corpus_profiler.py` | Full rewrite — all corrections below |
| `scripts/diagnostics/corpus_inventory.py` | Updated for new output schema |
| `tests/unit/test_corpus_profiler.py` | 61 tests (was 41); 20 new tests |
| `docs/evidence/memory/stage7_0a_corpus_inventory.md` | This corrected report |
| `runs/component_refinement/memory/00_corpus_inventory/` | Regenerated JSON (local only, not tracked) |

## Corrected formulas and boundary semantics

### Quantization (new)

All tie-sensitive operations round to **8 decimal places** before grouping.
This merges logically equal floating-point sums (e.g. `-1.2` and
`-1.2000000000000002`) into the same tie group.

```python
quantize(x) = round(x, 8)
```

### Factual return (corrected episode-end semantics)

```
G^H_t = sum(rewards[t : t+H])
```

Valid when `t+H <= T` and no `legacy_done` flag appears **before** the
final included position: `done[t : t+H-1]` must be all-`False`. A `done`
flag at position `t+H-1` (the final included reward) is **valid** —
the episode ended after collecting that reward. Horizon 1 is always valid
(the check slice is empty).

### Directional change (corrected episode-end semantics)

```
d_t(h) = mean(rewards[t:t+h]) - mean(rewards[t-h:t])
```

Valid when `t >= h`, `t+h <= T`, and no `done` in `[t-h, t+h-1]`.
A done flag at the final included position `t+h-1` is valid.

### Percentile ranks (quantized)

Values quantized to 8 decimals before tie grouping. Ties receive the mean
percentile rank of their group.

### Priority signals

```
qG_i   = percentile_rank(return_i, H)          # same H for both
q⁺_i   = max(0, 2*qG_i - 1)
q⁻_i   = max(0, 1 - 2*qG_i)
q↑_i   = positive_tail_percentile(d_i)         # using declared h
q↓_i   = positive_tail_percentile(-d_i)
```

`q_pos` and `q_neg` are **always** derived from the same `qG` (same H).
`q_up` and `q_down` are always derived from the same `d_t` (same h).

### Legacy done (corrected naming)

The rollout schema stores `done = terminated OR truncated` without
distinction. The field is named `legacy_done` to avoid claiming the
terminated/truncated distinction. No inference is made about episode-end
cause.

### Priority, crowding and uniform-mixture formula

```
score_i = λ⁺(eps+q⁺)^α + λ⁻(eps+q⁻)^α
        + λ↑(eps+q↑)^β + λ↓(eps+q↓)^β
        + λd * legacy_done_i

score'_i = score_i / count_same_return_i^rho
P(i) = eta/N + (1-eta) * score'_i / sum(score')
```

The selected candidates use `eta=0.1`, `rho=0.25`, and
`lambda_legacy_done=0`. Thus ten percent of total probability remains uniform;
the floor does not accumulate independently for every duplicate.

### Active-set simulation (new)

Weighted reservoir without replacement:

```
key_i = -log(U_i) / weight_i
keep M smallest keys
```

`U_i` is hash-derived from `(cycle_seed, stable_pointer_id)`. Inserting a new
pointer therefore does not rerandomize existing pointers. One hundred cycle
seeds are simulated per candidate.

## Targeted and full test results

```
$ pytest tests/unit/test_corpus_profiler.py -v
68 passed
```

New test coverage:
- `test_q_pos_q_neg_from_same_qG`: proves q⁺ and q⁻ share the same qG
- `test_numerically_close_tied_by_quantization`: floating artefacts merge
- `test_numerically_close_sums_deterministic_ties`: all-identical rewards
  produce one unique quantized value
- `test_done_at_final_included_is_valid`, `test_done_before_final_is_invalid`,
  `test_done_only_at_last_h4`: corrected episode-end semantics
- `test_directional_change_allows_done_at_end`,
  `test_directional_change_rejects_mid_window_done`: same for d_t
- `test_factual_return_quantized`: quantized sum equals -1.2 exactly
- `TestReservoirSampling` (10 tests): determinism, stable insertion,
  replacement, uniformity, identity validation and zero/NaN rejection
- `TestDensityMetrics` (5 tests): uniform, skewed, low/high-return-dense,
  single-dominant distributions
- `test_selected_horizons_persisted`: selected_H and selected_h in output

Full suite: `pytest tests/unit/` → 738 passed (no regressions).

## Corrected seed 42/43 corpus summary

| Metric | Seed 42 | Seed 43 |
|--------|:-------:|:-------:|
| Train files | 16 | 16 |
| Validation files | 4 | 4 |
| Transitions | 4280 | 4546 |
| Eligible pointers | 4280 | 4546 |
| Train/val disjoint | Yes | Yes |
| `legacy_done=True` | 0 | 0 |
| Selected H | 12 | 12 |
| Selected h | 4 | 4 |
| Quantize decimals | 8 | 8 |
| Gini weight (balanced + ρ=0.25) | 0.477 | 0.481 |

The `legacy_done` flag is always `False`. Episode-end cause is unavailable;
these files therefore cannot be classified as terminated or truncated.

## H sensitivity

Running the full sensitivity grid for each return horizon (`h=4` fixed):

| H | Seed-42 ESS/N | Seed-43 ESS/N |
|---|:-------------:|:-------------:|
| 1 | 0.446 | 0.447 |
| 2 | 0.487 | 0.488 |
| 4 | 0.578 | 0.580 |
| 8 | 0.699 | 0.695 |
| **12** | **0.746** | **0.744** |

H=12 has the most distinct factual-return values and matches the longest
currently validated blind horizon. Its higher ESS only means that its resulting
weights are less concentrated; ESS alone is not evidence of better memory.
**H=12 remains the proposed value, pending the replay comparison.**

## h sensitivity

Running the full sensitivity grid for each d_horizon (H=12 fixed):

| h | Seed-42 ESS/N | Seed-43 ESS/N | Seed-42 up/down events |
|---|:-------------:|:-------------:|:--------------------------:|
| 2 | 0.740 | 0.734 | 423 / 441 |
| 3 | 0.739 | 0.734 | 553 / 556 |
| **4** | **0.746** | **0.744** | **625 / 644** |

All candidates have similar ESS. `h=4` is retained because it smooths the
sparse reward pulses most strongly and costs very few eligible boundary
positions in this corpus. This is a factual-priority definition, not a claim
that `h=4` is optimal for policy learning.

## Active-set simulation (corrected)

For each candidate M, 100 cycle seeds of weighted reservoir sampling:

| Config | M | Replaced/cycle seed 42 | Replaced/cycle seed 43 | Unique ptrs |
|--------|---|:----------------------:|:----------------------:|:-----------:|
| uniform | 512 | 0.884 | 0.891 | ~100% |
| uniform | 1024 | 0.762 | 0.777 | 100% |
| uniform | 2048 | 0.521 | 0.550 | 100% |
| return_only | 512 | 0.868 | 0.876 | ≥99% |
| return_only | 1024 | 0.738 | 0.753 | 100% |
| return_only | 2048 | 0.491 | 0.519 | 100% |
| balanced | 512 | 0.846 | 0.855 | ≥99% |
| balanced | 1024 | 0.708 | 0.723 | 100% |
| balanced | 2048 | 0.461 | 0.489 | 100% |
| balanced + ρ=0.25 | 512 | 0.775 | 0.783 | ≥98% |
| balanced + ρ=0.25 | 1024 | 0.604 | 0.620 | ≥99% |
| balanced + ρ=0.25 | 2048 | 0.382 | 0.402 | 100% |

Key observations:
- Replacement is substantial at every M; it is now reported as
  `1 - |A∩B|/M`, separately from Jaccard distance.
- Nearly all unique pointers appear across 100 cycles even at M=512
- Tag composition is stable across M and configs (~37% each for positive/negative extremeness)
- Random keys are derived from `(cycle_seed, stable_pointer_id)`, so inserting a
  new record does not rerandomize existing records.

## Corrected density/concentration metrics (balanced config)

| Metric | Seed 42 | Seed 43 |
|--------|:-------:|:-------:|
| Highest-weight 10% mass | 0.350 | 0.355 |
| Lowest-return 10% weight share | 0.041 | 0.040 |
| Highest-return 10% weight share | 0.304 | 0.307 |
| Largest equal-return group pointer share | 0.646 | 0.651 |
| Largest equal-return group weight share | 0.263 | 0.264 |
| Gini (weight) | 0.477 | 0.481 |

The selected mild crowding correction reduces the largest equal-return group
from ~65% of pointers to ~26% of probability. It intentionally increases
priority contrast; the stronger `ρ=0.5` alternative was rejected because it
reduced the group to ~14% while collapsing ESS/N to ~0.26.

| ρ | Mean ESS/N | Dominant-group weight | Interpretation |
|---|:----------:|:---------------------:|----------------|
| 0 | 0.725 | 46% | Too much duplicate mass |
| **0.25** | **0.460** | **26%** | Selected compromise |
| 0.5 | 0.260 | 14% | Excessive rare-return concentration |
| 1.0 | 0.044 | 7% | Rejected collapse |

## Recommendations (all PENDING SUPERVISOR APPROVAL)

| Parameter | Recommendation | Rationale |
|-----------|---------------|-----------|
| Return horizon H | **12** | Most distinct returns; matches validated blind horizon |
| Smoothing h | **4** | Strongest local smoothing; negligible boundary cost |
| Uniform floor η | **0.1** | Preserves coverage while retaining priority contrast |
| λ⁺/λ⁻/λ↑/λ↓ | **1.0** | Equal weight across all four signal components |
| λ_legacy_done | **0** | No `done=True` samples in current corpus |
| α (return exp) | **1.0** | Linear; avoids extreme skew from tied returns |
| β (surprise exp) | **1.0** | Linear |
| Crowding ρ | **0.25** | Reduces the 65% duplicate group to 26% probability without the ρ=0.5 ESS collapse |
| Active-set M | **1024** | ~61% replacement/cycle, near-complete long-run coverage and half the materialization of M=2048 |
| Archive size | **All 4–5k pointers** | Cheap to retain full factual index |

## Git artifact cleanup

Generated `runs/` artifacts have been removed from Git tracking via
`git rm --cached -r runs/component_refinement/memory/`. Local files
remain on disk for future reference.

## Remaining ambiguities and blockers

1. **Terminated/truncated unavailable**: The rollout schema stores
   `done = terminated OR truncated` without separation, and all stored flags
   are false. Episode-ending cause is unknown. The `legacy_done` term is inert
   for this corpus.

2. **Tie compression**: 65% of pointers share the same quantized H=12
   return value (`-1.2`). This compresses the return-percentile resolution
   for ordinary transitions but does not break the priority formula, which
   spreads weight via the floor eta and surprise terms.

3. **Initial behavior coverage**: The corpus contains random and
   random-smooth collection variants, not a trained policy. Return and surprise
   distributions may shift after Stage 7.4 collection.

4. **No `legacy_done=True`**: The corpus has no episode-end transitions.
   The `lambda_legacy_done` coefficient cannot be validated. This is not
   a blocker for the main memory experiment but should be addressed when
   terminated episodes are collected.
