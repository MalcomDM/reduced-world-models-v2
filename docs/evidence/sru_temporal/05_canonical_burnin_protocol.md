# S4B — Canonical Visible Temporal / Burn-In Comparison

## Purpose

Compare causal Transformer, SRU burn-in 20, 8, 4, and 0 under identical
conditions. This is an apples-to-apples quality and efficiency study only.
Architecture selection is deferred to S6.

## Fixed Protocol

| Parameter | Value |
|-----------|-------|
| Temporal backends | `causal_transformer`, `minimal_sru` |
| SRU burn-in | 20, 8, 4, 0 |
| SRU training mode | `random_burn_in` (never sequential or macroblock) |
| Beta | 0.1 |
| Selection | learned Top-K, K=8 |
| Reward head | linear |
| Tokenizer eval mode | mean |
| Batch size | 8 |
| Sequence target length | 16 |
| Epochs | 10 |
| Observational dropout | disabled |
| Validation windows | 256 (held-out, same seed split) |
| Cache | `data/cache/rollout_frames_v1` |
| Evaluation | frozen-checkpoint evaluator + action probe |

## Variants

| ID | Backend | Burn-in | Seeds |
|----|---------|:-------:|:-----:|
| causal | causal_transformer | — | 42, 43 |
| sru20 | minimal_sru | 20 | 42, 43 |
| sru8 | minimal_sru | 8 | 42, 43 |
| sru4 | minimal_sru | 4 | 42, 43 |
| sru0 | minimal_sru | 0 | 42, 43 |

## Selection Tolerance

The smallest recurrent burn-in that:

1. Beats the constant-mean baseline in **both** seeds (visible val ratio < 1.0).
2. Remains within **0.02 ratio points** of SRU-20's mean frozen-evaluator ratio
   (averaged over both seeds).

Architecture selection remains deferred to S6.

## Commands

Each run uses the same skeleton:

```bash
python scripts/evaluate_reward_prediction.py \
  --out runs/component_refinement/sru_temporal/05_canonical_burnin_comparison/{ID}_seed{SEED} \
  --seed {SEED} --beta 0.1 --epochs 10 --batch-size 8 --max-val-windows 256 \
  --cache-dir data/cache/rollout_frames_v1 --reward-head-kind linear \
  --selection-mode learned --selection-k 8 --tokenizer-eval-mode mean \
  --temporal-backend {BACKEND} \
  {--sru-burn-in-steps BURNIN}
```

Explicit list:

```bash
# causal seed 42
python scripts/evaluate_reward_prediction.py --out .../causal_seed42 \
  --seed 42 ... --temporal-backend causal_transformer

# causal seed 43
python scripts/evaluate_reward_prediction.py --out .../causal_seed43 \
  --seed 43 ... --temporal-backend causal_transformer

# sru20 seed 42 (re-run of S2 visible anchor protocol)
python scripts/evaluate_reward_prediction.py --out .../sru20_seed42 \
  --seed 42 ... --temporal-backend minimal_sru --sru-burn-in-steps 20

# sru20 seed 43
python scripts/evaluate_reward_prediction.py --out .../sru20_seed43 \
  --seed 43 ... --temporal-backend minimal_sru --sru-burn-in-steps 20

# sru8 seed 42
python scripts/evaluate_reward_prediction.py --out .../sru8_seed42 \
  --seed 42 ... --temporal-backend minimal_sru --sru-burn-in-steps 8

# sru8 seed 43
python scripts/evaluate_reward_prediction.py --out .../sru8_seed43 \
  --seed 43 ... --temporal-backend minimal_sru --sru-burn-in-steps 8

# sru4 seed 42
python scripts/evaluate_reward_prediction.py --out .../sru4_seed42 \
  --seed 42 ... --temporal-backend minimal_sru --sru-burn-in-steps 4

# sru4 seed 43
python scripts/evaluate_reward_prediction.py --out .../sru4_seed43 \
  --seed 43 ... --temporal-backend minimal_sru --sru-burn-in-steps 4

# sru0 seed 42
python scripts/evaluate_reward_prediction.py --out .../sru0_seed42 \
  --seed 42 ... --temporal-backend minimal_sru --sru-burn-in-steps 0

# sru0 seed 43
python scripts/evaluate_reward_prediction.py --out .../sru0_seed43 \
  --seed 43 ... --temporal-backend minimal_sru --sru-burn-in-steps 0
```

## Post-Run Evaluation

After all 10 runs complete:

1. Read `results.json` for best/final val MSE, peak GPU, elapsed time.
2. Read last row of `metrics.csv` for sequential metrics (opt_updates,
   real_target_transitions, processed_model_positions).
3. Run action probe from each best checkpoint.
4. Run frozen-checkpoint evaluator (same window set, tokenizer mean, seed 0)
   for the official comparison ratio.
5. Produce RESULTS.md with per-seed and cross-seed mean tables.

## Files Per Run

```
{ID}_seed{SEED}/
  config.json          # Resolved ExperimentConfig
  dataset_manifest.json
  metrics.csv          # Epoch-level metrics
  results.json         # Summary
  checkpoints/
    checkpoint_best.pt  # Best model
    checkpoint_latest.pt
  best_world_model.pt  # Legacy bare state_dict
```
