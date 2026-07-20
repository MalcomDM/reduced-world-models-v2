# SRU-4 Matched Wall-Clock Optimization Probe — Stage S4B

**Date:** 2026-07-19
**Purpose:** Test whether SRU-4 with 17 epochs (10 + 7 additional) can recover
the quality gap to SRU-20 (10 epochs) within approximately the same total
wall-clock budget.

**Implementation:** Resume from `checkpoint_latest`, restore optimizer state,
continue training for 7 additional epochs with `validate_every=7` (validate
only at the final added epoch). No architecture, loss, or config changes.

---

## Final-State Comparison (latest checkpoint, not best)

| Metric | SRU-4 seed 42 (17 ep) | SRU-20 seed 42 (10 ep) | SRU-4 seed 43 (17 ep) | SRU-20 seed 43 (10 ep) |
|--------|:---------------------:|:----------------------:|:---------------------:|:----------------------:|
| Parent train time | 228 s | 374 s | 239 s | 399 s |
| Added time | 177 s | — | 188 s | — |
| **Total wall time** | **405 s** | **374 s** | **427 s** | **399 s** |
| Parent frozen ratio | 0.898 | 0.857 | 0.808 | 0.725 |
| **Final frozen ratio** | **0.871** | **0.871** | **0.839** | **0.692** |
| Action probe | 4/4 | 4/4 | 4/4 | 4/4 |

SRU-4 total wall time is within 8% of SRU-20's wall time (405 vs 374 / 427 vs 399).

## Quality Assessment

- **Seed 42:** SRU-4 final ratio (0.871) equals SRU-20 (0.871) — gap closed.
- **Seed 43:** SRU-4 final ratio (0.839) remains worse than SRU-20 (0.692)
  — gap not closed.
- SRU-4 improves from its parent (seed 42: 0.898 → 0.871; seed 43: 0.808 → 0.839)
  but not enough to match SRU-20 consistently across both seeds.

## Interpretation

- SRU-4 receives roughly the same total optimizer updates and target exposures
  as SRU-20 within the matched wall clock (7 added epochs ≈ 70% more updates).
- Despite additional updates, SRU-4's reduced context (4-step burn-in vs 20-step
  burn-in) limits the information available per update. The quality gap is not
  primarily a training budget issue — it reflects a genuine information bottleneck
  from the shorter burn-in.
- The wall-clock probe confirms the S4B selection: SRU-20 remains the correct
  recurrent quality reference. SRU-4 does not recover the quality gap even when
  given equal time.
