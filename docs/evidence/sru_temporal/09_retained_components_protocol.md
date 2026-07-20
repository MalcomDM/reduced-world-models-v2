# S4C — Minimal Retained-Component Confirmation Protocol

## Purpose

Confirm that the retained perception defaults (learned K=8, tokenizer mean)
remain justified after replacing the causal Transformer with MinimalSRU.

## Canonical Baseline

All variants use the strict SRU-20 protocol:
- `minimal_sru`, `burn_in=20`, `seq_len=16`, `random_burn_in`
- Strict `pre_perception_skip` observational dropout
- Warmup=4, horizons 1/2/4/8/12, probability=0.5, ramp=2
- `beta=0.1`, linear reward head, batch size 8, 10 epochs
- Seeds 42 and 43
- Frame cache enabled
- Validation: 256 held-out windows, tokenizer `mean`
- Masked evaluation with correct/zero/shifted action controls

## Experiment Matrix

| Group | Variant | Variable | Seeds | Expected runs |
|-------|---------|----------|:-----:|:-------------:|
| A | Learned K=8 (baseline) | — | 42, 43 | Existing (not retrained) |
| B | Fixed-random K=8 | `selection_mode=fixed_random`, `selection_seed=123` | 42, 43 | 2 |
| C | Learned K=16 | `selection_k=16` | 42, 43 | 2 |
| D | Tokenizer mean vs sample | `--tokenizer-eval-mode` at eval time | 42, 43 | No retraining |

## Future Commands

```bash
BASE=runs/component_refinement/sru_temporal/09_retained_components

# Fixed-random K=8, both seeds
for seed in 42 43; do
  python scripts/evaluation/evaluate_reward_prediction.py \
    --out "$BASE/fixed_random_k8_seed${seed}" \
    --seed "$seed" --beta 0.1 --epochs 10 --batch-size 8 \
    --sequence-len 16 --max-val-windows 256 \
    --cache-dir data/cache/rollout_frames_v1 --reward-head-kind linear \
    --selection-mode fixed_random --selection-k 8 --selection-seed 123 \
    --tokenizer-eval-mode mean \
    --temporal-backend minimal_sru --sru-training-mode random_burn_in \
    --sru-burn-in-steps 20 \
    --temporal-mask-enabled --temporal-mask-warmup 4 \
    --temporal-mask-horizons 1 2 4 8 12 \
    --temporal-mask-probability 0.5 --temporal-mask-ramp-epochs 2 \
    --observation-dropout-execution pre_perception_skip
done

# Learned K=16, both seeds
for seed in 42 43; do
  python scripts/evaluation/evaluate_reward_prediction.py \
    --out "$BASE/learned_k16_seed${seed}" \
    --seed "$seed" --beta 0.1 --epochs 10 --batch-size 8 \
    --sequence-len 16 --max-val-windows 256 \
    --cache-dir data/cache/rollout_frames_v1 --reward-head-kind linear \
    --selection-mode learned --selection-k 16 \
    --tokenizer-eval-mode mean \
    --temporal-backend minimal_sru --sru-training-mode random_burn_in \
    --sru-burn-in-steps 20 \
    --temporal-mask-enabled --temporal-mask-warmup 4 \
    --temporal-mask-horizons 1 2 4 8 12 \
    --temporal-mask-probability 0.5 --temporal-mask-ramp-epochs 2 \
    --observation-dropout-execution pre_perception_skip
done
```

## Evaluation Protocol

Use `checkpoint_best.pt` for every gate comparison. `checkpoint_latest.pt` may
be inspected as a diagnostic, but it must not replace the best checkpoint.

For each best checkpoint:
1. Run canonical visible evaluation with tokenizer `mean`, using the checkpoint's
   data-split seed and a fixed inference RNG seed.
2. Run the corrected target-relative masked factual evaluator with tokenizer
   `mean` (correct/zero/shifted actions, H=1/2/4/8/12).
3. Run the action probe and require 4/4 distinct outputs.

For the existing learned-K=8 checkpoints only, compare tokenizer policies
without retraining:
- `mean`: inference RNG seeds 0 and 1; results must be exactly reproducible.
- `sample`: inference RNG seeds 42, 43, and 44.
- Keep the checkpoint's data split fixed in every comparison.
- Use canonical visible evaluation for this policy comparison. The corrected
  masked factual evaluator intentionally requires `mean` so action-history
  comparisons remain deterministic.

## Expected Artifacts

```
runs/component_refinement/sru_temporal/09_retained_components/
  PROTOCOL.md
  RESULTS.md
  fixed_random_k8_seed42/
  fixed_random_k8_seed43/
  learned_k16_seed42/
  learned_k16_seed43/
  evaluation/
```

## Predeclared Interpretation Rules

1. **Learned selection** remains supported if its cross-seed mean masked ratio
   is no worse than fixed-random. If the masked-ratio gap is at most 0.01 and
   learned wins canonical visible ratio in both seeds, record a tie with a
   visible-quality trade-off. A learned deficit greater than 0.01 is a failed
   retained-component gate.
2. **K=16** replaces K=8 only if it improves cross-seed mean masked ratio
   by more than 0.02. Otherwise retain K=8.
3. **Tokenizer mean** is retained if it is exactly reproducible and sampling
   does not improve the cross-seed mean canonical visible ratio by more than
   0.01. Otherwise flag the inference policy for review rather than silently
   changing the default.
