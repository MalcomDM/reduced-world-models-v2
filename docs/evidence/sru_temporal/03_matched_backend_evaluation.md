# Matched Frozen-Checkpoint Evaluation — Stage S2.6

**Date:** 2026-07-19
**Protocol:** All checkpoints evaluated with tokenizer `mean`, 256 held-out windows,
batch size 8, sequence length 16, cache `data/cache/rollout_frames_v1`, inference RNG seed 0.
No retraining, no observational masking.

| Backend | Training | Seed | MSE | MAE | Baseline MSE | Ratio | Action probe |
|---------|----------|:----:|:---:|:---:|:------------:|:-----:|:------------:|
| Causal Transformer | Transformer | 42 | 0.4137 | 0.3136 | 0.4785 | 0.8645 | 4/4 PASS |
| Causal Transformer | Transformer | 43 | 0.4661 | 0.2502 | 0.5982 | 0.7792 | 4/4 PASS |
| SRU random-burn-in | random-burn-in | 42 | 0.4015 | 0.2554 | 0.4785 | 0.8392 | 4/4 PASS |
| SRU random-burn-in | random-burn-in | 43 | 0.4553 | 0.2772 | 0.5982 | 0.7610 | 4/4 PASS |
| SRU macroblock M=64 | M=64 | 42 | 0.4422 | 0.3196 | 0.4785 | 0.9242 | 4/4 PASS |
| SRU macroblock M=64 | M=64 | 43 | 0.5015 | 0.2947 | 0.5982 | 0.8384 | 4/4 PASS |

### Interpretation

1. **Temporal architecture vs training regime:** Causal Transformer and SRU random-burn-in
   anchors were trained with the same 10-epoch, beta=0.1, K=8 protocol. SRU macroblock
   was trained with 151 passes at M=64. The evaluation protocol is identical for all.

2. **Cache invariance:** The frame cache changes runtime but cannot alter prediction values.
   All checkpoints were evaluated with the same cache.

3. **Posterior-mean evaluation:** Using `tokenizer_eval_mode=mean` (deterministic) does not
   change the ranking of the frozen anchors. All ratios are comparable within seed.

4. **No efficiency claim from historical times:** Causal, SRU burn-in, and SRU macroblock
   were trained under different cache/batch/epoch protocols. Elapsed times from training
   are not comparable across regimes.

### Conclusion

All six checkpoints produce valid held-out reward predictions with deterministic action
probes. No checkpoint/evaluator incompatibility was found. Stage S3 is authorized.
