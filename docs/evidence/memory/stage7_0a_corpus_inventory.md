# Stage 7.0A — Factual Memory Corpus Inventory

## Summary

A read-only profiler over eligible training-file transitions for Stage 7
memory. Produces per-pointer metrics, corpus diagnostics, and a sensitivity
grid for continuous priority weights.

## Files changed

| File | Purpose |
|------|---------|
| `src/rwm/memory/__init__.py` | Package init |
| `src/rwm/memory/corpus_profiler.py` | Core analysis: returns, directional change, percentile ranks, weights, ESS, correlations, dense-region impact |
| `scripts/diagnostics/corpus_inventory.py` | Thin CLI; writes JSON per seed, RESULTS.md, RUN_INDEX.md |
| `tests/unit/test_corpus_profiler.py` | 41 tests covering all requirements |
| `runs/component_refinement/memory/00_corpus_inventory/seed42/corpus_summary.json` | Seed-42 raw profile |
| `runs/component_refinement/memory/00_corpus_inventory/seed43/corpus_summary.json` | Seed-43 raw profile |
| `runs/component_refinement/memory/00_corpus_inventory/RESULTS.md` | Auto-generated summary |
| `runs/component_refinement/memory/RUN_INDEX.md` | Stage-7 run index |
| `docs/evidence/memory/stage7_0a_corpus_inventory.md` | This document |

## Exact formulas and boundary semantics

### Factual return (undiscounted)

```
G^H_t = sum(rewards[t : t+H])
```

Valid when `t+H <= T` and no `done` flag appears in `[t, t+H-1]`. Invalid
pointers receive NaN. Horizon 1 is always valid.

### Directional reward-rate change

```
d_t(h) = mean(rewards[t:t+h]) - mean(rewards[t-h:t])
```

Valid when `t >= h`, `t+h <= T`, and no `done` in `[t-h, t+h-1]`.

### Percentile ranks

```
rank = mean rank of tied group (ties averaged)
pct = rank / (n - 1)    for n > 1
pct = 0.5               for n == 1
```

Tied values receive the mean percentile rank of their group.

### Derived priority signals

```
qG_i   = percentile_rank(return_i, H)          # factual return rank
q⁺_i   = max(0, 2*qG_i - 1)                   # upper-half extremeness
q⁻_i   = max(0, 1 - 2*qG_i)                   # lower-half extremeness
q↑_i   = positive_tail_percentile(d_i)         # upward surprise
q↓_i   = positive_tail_percentile(-d_i)        # downward surprise
```

`positive_tail_percentile`: for values > 0, percentile rank among
strictly positive values; zero otherwise.

### Weight formula

```
w_i = eta
    + λ⁺ (ε + q⁺_i)^α
    + λ⁻ (ε + q⁻_i)^α
    + λ↑ (ε + q↑_i)^β
    + λ↓ (ε + q↓_i)^β
    + λT * terminal_i
```

where ε = 1e-6, `terminal_i` is 1 if `done[i]==True` else 0.

### Effective sample size

```
ESS = (sum w_i)^2 / sum(w_i^2)
```

## Corpus summary (seeds 42 and 43)

### Overall

| Metric | Seed 42 | Seed 43 |
|--------|:-------:|:-------:|
| Train files | 16 | 16 |
| Validation files | 4 | 4 |
| Total transitions | 4280 | 4546 |
| Eligible pointers | 4280 | 4546 |
| Train/val disjoint | Yes | Yes |
| Any `done=True` | No | No |
| Episode lengths | 148–845 | 148–845 |

All episodes are truncated (timed out). No terminated transitions exist in
the current corpus. The `terminated_OR_truncated` flag is always `False`.

### Reward distribution

| Stat | Value |
|------|-------|
| Min reward | -0.1 |
| Median reward | -0.1 |
| Mean reward | ~0.078 |
| Max reward | ~10.8 |
| Std | ~0.82–0.84 |
| Unique values | 67–69 |

Rewards are highly skewed: ~75% are exactly -0.1 (off-road penalty), with
rare positive spikes up to 10.8 (track-progress reward).

### Return quantiles

| Horizon | Min | Median | Mean | Max | Std | Unique vals |
|---------|-----|--------|------|-----|-----|-------------|
| H=1 | -0.1 | -0.1 | 0.08 | 10.8 | 0.84 | 67 |
| H=4 | -0.4 | -0.4 | 0.23 | 10.5 | 1.44 | 120 |
| H=12 | -1.2 | -1.2 | 0.72 | 16.8 | 3.09 | 305 |

All medians equal the minimum value for each horizon: the most common
return is pure off-road (all -0.1 rewards). Longer horizons produce more
unique values (305 for H=12 vs 67 for H=1). **H=12 is recommended** for
meaningful separation.

### Tie analysis

>96% of values are tied at every horizon. The largest tie at H=12 covers
1078 pointers (26%). This means the bottom ~50% of the return distribution
collapses into a single percentile rank ~0.27, producing **no pointers in
the 0–25% percentile rank bucket**. A crowding correction may be warranted.

### Directional change (surprise signals)

| h | Up count (%) | Down count (%) | Mean d_t | Std d_t |
|---|:-----------:|:-------------:|:--------:|:-------:|
| 2 | 423 (9.9%) | 441 (10.3%) | -0.014 | 0.75 |
| 3 | 553 (12.9%) | 556 (13.0%) | -0.009 | 0.56 |
| 4 | 625 (14.6%) | 644 (15.0%) | -0.007 | 0.44 |

h=4 captures the most surprise events with the lowest noise (std 0.44).
**h=4 is recommended** for the priority formula d_t component.

### Signal correlations

- `return_H=1` vs `q_neg`: **-0.96** (near-perfect — short returns are dominated by
  the -0.1 penalty, making any positive return immediately "negative-priority")
- `return_H=12` vs `q_pos`: **0.95** (long-horizon returns align closely with
  upper-tail extremeness)
- Adjacent horizons correlate strongly: `H=8 vs H=12`: 0.91;
  `h=3 vs h=4`: 0.80
- `return_H=2` vs `q_up_h=2`: **0.84** (short-horizon return spikes are detected
  by short-window reward-rate change)

The `return_H=1` ↔ `q_neg` correlation indicates that q_neg and q_pos can
partially substitute for return percentile. No signal pair is redundant
enough to drop.

## Sensitivity grid

| Config | η | λ⁺/λ⁻ | λ↑/λ↓ | λT | α/β | ESS (seed42) | ESS ratio |
|--------|---|---|---|---|-----|:-----------:|:---------:|
| uniform | 1.0 | 0/0 | 0/0 | 0 | 1/1 | 4280 | 1.00 |
| return_only | 0.1 | 1/1 | 0/0 | 0 | 1/1 | 2384 | 0.56 |
| return_sharp | 0.1 | 2/2 | 0/0 | 0 | 2/1 | 1617 | 0.38 |
| change_focused | 0.1 | 0.5/0.5 | 2/2 | 1 | 1/2 | 1644 | 0.38 |
| **balanced** | **0.1** | **1/1** | **1/1** | **1** | **1/1** | **2079** | **0.49** |
| high_floor | 0.5 | 0.5/0.5 | 0.5/0.5 | 0.5 | 1/1 | 3778 | 0.88 |
| return_extreme | 0.1 | 2/2 | 0.5/0.5 | 1 | 2/1.5 | 1636 | 0.38 |

Seed 43 results are essentially identical (ESS ratios within 0.01).

The **balanced** config is recommended as the default: it gives moderate
ESS reduction (~50% of uniform) while allocating weight across all four
signal components. The **high_floor** config is the safest if diversity is
the primary concern.

## Dense-region impact (balanced config)

| Stat | Value |
|------|-------|
| Top-10% weight fraction | 3.6–3.9% |
| Top-25% weight fraction | 9.5–9.7% |
| Top-50% weight fraction | 19.0–19.5% |
| Gini coefficient | 0.50 |

Weight is well-spread across the return distribution. No single return
region dominates. The crowding correction is not urgently needed but may
help separate the compressed low-return region.

## Candidate-configuration recommendations

All **PENDING SUPERVISOR APPROVAL**:

| Parameter | Recommendation | Rationale |
|-----------|---------------|-----------|
| Return horizon H | **12** | Most unique values (305), best separation |
| Smoothing h | **4** | Most surprise events, lowest std |
| Uniform floor η | **0.1** | Balances diversity vs. prioritization |
| λ⁺ (positive) | **1.0** | Equal weight to signal components |
| λ⁻ (negative) | **1.0** | Equal weight |
| λ↑ (upward surprise) | **1.0** | Equal weight |
| λ↓ (downward surprise) | **1.0** | Equal weight |
| λT (terminal) | **1.0** | (currently unused — all done=False) |
| α (return exponent) | **1.0** | Linear — avoids extreme skew from tied returns |
| β (surprise exponent) | **1.0** | Linear |
| Active-set size M | **all pointers** | Only 4–5k pointers; cheap to keep all |
| Crowding correction | **Not yet justified** | Gini=0.50, no single region dominant |

## Ambiguities and blockers

1. **Terminated/truncated distinction**: The rollout schema stores
   `done = terminated or truncated` without separation. All transitions
   are truncated (timed out). The `λT terminal_i` term is currently 0 for
   all pointers. A schema extension is needed before terminal-aware
   priorities matter.

2. **Tie compression in low-return region**: The minimum return (-1.2 at
   H=12) accounts for >50% of pointers. All these receive the same
   percentile rank (~0.27), making the bottom quartile of the return
   distribution indistinguishable. This does not break the priority formula,
   but it means q⁺/q⁻ signals are partially degenerate for ordinary
   transitions. The crowding correction (`gap_i * w_i`) is the planned
   mitigation.

3. **Signal redundancy**: `return_H=1` and `q_neg` are near-perfectly
   anti-correlated (-0.96). Using H=12 for qG reduces this overlap, but
   q_neg and H=1 returns remain conceptually linked. The current formula
   uses H=12 returns for qG, which mitigates this.

4. **No `done=True` events**: The entire corpus is truncated without
   termination. Terminal-priority weighting cannot be validated until
   terminated episodes are collected or synthetic done events are
   introduced. This is not a blocker for the main memory experiment, but
   the terminal term should be evaluated once available.
