# Strict Observational-Dropout Protocol — S4C.1

## Purpose

Validate that MinimalSRU learns a useful recurrent dynamic state when masked
observations truly bypass the entire perception stack.

## Variants

| Regime | Execution policy | Description |
|--------|-----------------|-------------|
| Visible-only SRU-20 | — | Reference: no masking |
| Post-perception masked | `post_perception` | Mask after CNN (existing) |
| Strict pre-perception masked | `pre_perception_skip` | Bypass perception entirely (new) |

## Commands

```bash
# Seed 42
python scripts/evaluation/evaluate_reward_prediction.py \
  --out runs/.../08_strict_observational_dropout_anchor/seed42 \
  --seed 42 --beta 0.1 --epochs 10 --batch-size 8 \
  --sequence-len 16 --max-val-windows 256 \
  --cache-dir data/cache/rollout_frames_v1 \
  --reward-head-kind linear --selection-mode learned --selection-k 8 \
  --tokenizer-eval-mode mean \
  --temporal-backend minimal_sru --sru-burn-in-steps 20 \
  --temporal-mask-enabled --temporal-mask-warmup 4 \
  --temporal-mask-horizons 1 2 4 8 12 \
  --temporal-mask-probability 0.5 --temporal-mask-ramp-epochs 2 \
  --observation-dropout-execution pre_perception_skip

# Seed 43: same command with --seed 43
```

## Evaluation

For each checkpoint:
1. Visible canonical reward prediction (256 windows, tokenizer mean)
2. Masked factual horizons H={1,2,4,8,12}, anchored to the first
   `loss_mask=True` target after the 20-position burn-in, with
   correct/zero/shifted previous-action histories
3. Action sensitivity probe

The runtime execution policy is read from each checkpoint: strict checkpoints
use `pre_perception_skip`; controls use `post_perception`. Every masked result
records exact perceived/skipped valid positions and uses the full training
split for the constant-reward baseline.
