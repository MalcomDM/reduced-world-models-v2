# SRU Macroblock M=64 Matched-Exposure — Stage S2.5B Results

**Date:** 2026-07-19
**Config:** `backend=minimal_sru`, `sru_training_mode=random_macroblock_tbptt`,
`burn_in=20`, `macroblock_target_steps=64`, `tbptt_steps=16`, `beta=0.1`,
K=8 learned, linear reward head, tokenizer `mean`, no masking.
**Matched reference:** SRU Stage S2 random-burn-in anchors under the same
backend and visible-reward objective (seed 42 ratio 0.854; seed 43 ratio
0.791). The older causal-Transformer lineage is a separate architectural
reference and is not used for this S2.5 comparison.

---

## Per-Seed Results

| Metric | Seed 42 | Seed 43 |
|--------|:-------:|:-------:|
| Best val MSE | **0.4907** | **0.5368** |
| Baseline MSE | 0.5629 | 0.6283 |
| **Ratio** | **0.872** | **0.854** |
| Final val MAE | 0.2645 | 0.3145 |
| Training time | 259.6 s | 269.2 s |
| Optimizer updates (cumulative) | 6,040 | 6,040 |
| Macro passes | 151 | 151 |
| Real target transitions | 4,280 | 4,546 |
| Real burn-in transitions | 1,180 | 1,240 |
| Processed model positions | 6,300 | 6,552 |
| Peak GPU | 0.81 GB | 0.81 GB |
| Action probe (4/4) | ✅ | ✅ |

The best seed-42 checkpoint occurred at macro pass 60 (rather than the final
151); seed 43 improved through the final pass. The sparse validation curve is
therefore evidence of optimization variability, not a monotonic convergence
claim.

### Cross-seed mean ratio

**Macroblock mean: (0.872 + 0.854) / 2 = 0.863**
**S2 random-burn-in mean: (0.854 + 0.791) / 2 = 0.822**

---

## Comparison with S2 Random-Burn-In Anchors

| Dimension | Macroblock M=64 (mean) | S2 Random-burn-in (mean) |
|-----------|:---------------------:|:------------------------:|
| Direct supervised targets | 666,363 | 667,520 |
| Processed model positions | 970,326 | 1,501,920 |
| Optimizer updates | 6,040 | 5,215 |
| Wall time | 264 s | 386 s |
| GPU memory | 0.81 GB | 1.23 GB |

### Interpretation

- **Target exposure is matched by construction.** The metrics stored per row
  describe one macro pass and must be accumulated across all 151 passes:
  seed 42: ``4,280 × 151 = 646,280`` targets (0.02% below the S2 reference
  646,400); seed 43: ``4,546 × 151 = 686,446`` (0.32% below 688,640).

- The macro protocol has **more**, not fewer, optimizer updates: 6,040 versus
  5,050/5,380 in the matched S2 references. It processes fewer model positions
  because each 20-step burn-in is shared across 64 targets rather than 16.

- **GPU memory is lower** (0.81 GB vs 1.23 GB). At matched target exposure,
  macroblocks process about 35--36% fewer model positions than S2.

- **Ratio is worse** (0.863 vs 0.822 mean). Thus this first M=64 configuration
  buys lower model-position cost, memory, and measured wall time, but does not
  yet preserve the random-burn-in reward-prediction quality.

- **Not a comparison of nominal epochs:** the controlled budget is direct target
  exposure; optimizer updates and processed positions are reported separately.

---

## Statements

This experiment measures world-model reward learning and state-refresh efficiency
only. It does not prove actor-critic stability, imagined-rollout quality, or
end-to-end training safety.
