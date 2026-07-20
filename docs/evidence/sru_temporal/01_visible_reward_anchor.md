# SRU Visible Reward Anchor — Stage S2 Results

**Date:** 2026-07-19
**Causal baseline branch:** `baseline/causal-transformer-stage5`
**SRU config:** `backend=minimal_sru`, 20-step burn-in + 16 target, `beta=0.1`,
K=8 learned, linear reward head, tokenizer `mean`, no observation masking.
**Dataset:** `data/rollouts/rwm_deterministic/scenario_0` (20 episodes, 5339 steps)

---

## Metrics

| Metric | SRU seed 42 | SRU seed 43 | Causal reference (seed 42) |
|--------|:-----------:|:-----------:|:--------------------------:|
| Best val MSE | **0.4807** | **0.4964** | 0.4663 |
| Best val MAE | — | — | — |
| Baseline MSE | 0.5630 | 0.6279 | — |
| **Ratio (MSE/baseline)** | **0.854** | **0.791** | **0.828** |
| Temporal params | 5,920 | 5,920 | 56,560 |
| Total model params | ~42,884 | ~42,884 | ~93,524 |
| Total wall time | 373.5s | 398.6s | 306.6s† |
| Epoch time | ~37s | ~38s | ~30.7s† |
| Peak GPU | 1.23 GB | 1.23 GB | ~0.56 GB† |
| Action probe (4/4) | ✅ | ✅ | ✅ |
| Checkpoint backend | `minimal_sru` | `minimal_sru` | `causal_transformer` |

*SRU processes 36-frame sequences (burn-in+target) vs causal 16-frame windows,
explaining higher GPU memory.

†The causal quality reference is the vectorized Stage 02 anchor (same reported
MSE/ratio source).  It was uncached.  Its older 54 s/epoch number belongs to
the pre-vectorization Stage 01 run and must not be paired with Stage 02 quality.
The cached causal K=8 ablation ran in ~15.6 s/epoch, but is not an identical
quality/configuration reference.  Therefore this table reports no speed winner.

### Seed mean ratio

**SRU mean ratio: (0.854 + 0.791) / 2 = 0.822**

Causal mean ratio (Stage 02 vectorized reference):
``(0.828 + 0.787) / 2 = 0.808``.

Both SRU seeds beat the constant-mean baseline. SRU seed 43 is competitive
with the causal reference (0.791 vs 0.767 causal seed 43, 0.791 vs 0.828
causal seed 42). SRU seed 42 is slightly above the causal seed 42 reference
(0.854 vs 0.828).

---

## Interpretation

1. **Both SRU seeds beat the mean baseline.** This satisfies the S2 visible
   anchor gate.

2. **SRU is not capacity-limited.** At 5,920 temporal parameters (vs 56,560
   for the Transformer), the SRU achieves comparable held-out reward
   prediction.

3. **Runtime is not yet a fair architecture verdict.** SRU took ~37–38 s per
   epoch versus ~30.7 s for the uncached vectorized causal quality reference.
   The older ~54 s causal figure was pre-vectorization and is not comparable.
   A matched cached causal/SRU throughput benchmark is required before making
   an efficiency claim.

4. **Memory is higher** (1.23 GB vs ~0.5 GB) because SRU processes 36 frames
   per window vs 16 for causal.

5. **No excessive KL** — KL is supervised only on 16 target positions, not
   the full 36-frame sequence. Direct comparison with causal KL magnitudes
   is not meaningful.

---

## Gate Decisions

| Gate | Status |
|------|--------|
| Both seeds beat constant-mean baseline | ✅ Pass |
| Action probe 4/4 | ✅ Pass |
| Checkpoint reloads with correct backend | ✅ Pass |
| Cross-seed mean ratio > 0.05 regression? | No — mean ratio 0.822 vs causal 0.808 (span of ~0.014) |
| **S3 authorized?** | **Yes** |
