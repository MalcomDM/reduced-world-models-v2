# Strict Observational-Dropout Anchor — Corrected Evaluation

**Date:** 2026-07-19  
**Status:** PASS after target-relative evaluator correction  
**Training:** MinimalSRU, burn-in 20, target length 16, 10 epochs, K=8,
beta=0.1, tokenizer mean, temporal masking with 4 visible target steps and
blind horizons 1/2/4/8/12.

## Evaluation correction

The first evaluation incorrectly counted `warmup=4` from layout position zero,
inside the 20-position burn-in. The corrected evaluator:

- keeps every valid burn-in position visible;
- counts warmup from the first `loss_mask=True` target;
- masks and scores exactly positions `20 + 4 : 20 + 4 + H`;
- excludes `valid_step=False` padding;
- uses the full training split for the constant-reward baseline;
- forwards the checkpoint's explicit dropout execution policy;
- records exact perceived/skipped valid positions.

No checkpoint was retrained.

## Canonical visible reward quality

| Regime | Seed 42 ratio | Seed 43 ratio | Mean |
|---|---:|---:|---:|
| Visible-only SRU-20 | 0.837 | 0.635 | 0.736 |
| Post-perception masked | 0.836 | 0.773 | 0.805 |
| **Strict pre-perception masked** | **0.887** | **0.814** | **0.851** |

Both strict checkpoints remain below the constant-mean baseline. Strict
training sacrifices some visible-only accuracy, especially relative to the
visible-only seed-43 run, but preserves useful visible reward prediction.

## Corrected masked factual ratios — correct action history

| Seed | Regime | H=1 | H=2 | H=4 | H=8 | H=12 | Mean |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | Visible-only SRU-20 | 0.922 | 0.948 | 0.955 | 0.969 | 0.978 | 0.954 |
| 42 | Post-perception masked | 0.917 | 0.946 | 0.895 | 0.905 | 0.921 | 0.917 |
| 42 | **Strict pre-perception** | **0.915** | **0.928** | **0.907** | **0.918** | **0.935** | **0.920** |
| 43 | Visible-only SRU-20 | 0.991 | 1.020 | 1.021 | 1.037 | 1.049 | 1.023 |
| 43 | Post-perception masked | 0.916 | 0.924 | 0.936 | 0.945 | 0.952 | 0.935 |
| 43 | **Strict pre-perception** | **0.965** | **0.951** | **0.947** | **0.950** | **0.955** | **0.953** |

Strict masked training improves every matched horizon over the visible-only
control in both seeds. All strict masked ratios remain below 1.0, including
H=12.

## Strict action-history controls

| Seed | H | Correct | Zero | Shifted |
|---:|---:|---:|---:|---:|
| 42 | 1 | 0.915 | **0.910** | 0.911 |
| 42 | 2 | 0.928 | **0.925** | 0.928 |
| 42 | 4 | **0.907** | 0.915 | 0.907 |
| 42 | 8 | **0.918** | 0.941 | 0.918 |
| 42 | 12 | **0.935** | 0.965 | 0.935 |
| 43 | 1 | 0.965 | 0.966 | **0.958** |
| 43 | 2 | 0.951 | 0.953 | **0.948** |
| 43 | 4 | 0.947 | 0.951 | **0.946** |
| 43 | 8 | 0.950 | 0.954 | **0.950** |
| 43 | 12 | **0.955** | 0.959 | 0.957 |

Removing previous-action history generally hurts at longer horizons, most
clearly in seed 42. Correct versus one-step-shifted history is marginal and
mixed, so this experiment supports action-history usefulness but does not
establish precise action-index identification by itself. The separate 4/4
probe establishes current-action conditioning of the reward head.

## Strict versus post-perception

Mean correct-history masked ratios:

| Seed | Post-perception | Strict | Strict − post |
|---:|---:|---:|---:|
| 42 | 0.917 | 0.920 | +0.004 |
| 43 | 0.935 | 0.953 | +0.019 |

Strict execution stays comfortably within the predeclared +0.05 tolerance.
The two policies are close in reward quality; strict is slightly worse on
average.

## Exact mask accounting

For 256 windows, every horizon scores exactly `256 × H` transitions. Valid
burn-in length varies only for episode-start windows, where left padding is
excluded:

| H | Scored/skipped | Perceived valid seed 42 | Perceived valid seed 43 |
|---:|---:|---:|---:|
| 1 | 256 | 8,765 | 8,619 |
| 2 | 512 | 8,509 | 8,363 |
| 4 | 1,024 | 7,997 | 7,851 |
| 8 | 2,048 | 6,973 | 6,827 |
| 12 | 3,072 | 5,949 | 5,803 |

The full recurrent input has 20 context plus 16 target positions. Every valid
burn-in position and the first four target positions remain visible.

## Runtime and scope

| Regime | Seed 42 train time | Seed 43 train time | Peak GPU |
|---|---:|---:|---:|
| Visible-only | 374 s | 399 s | 1.23 GB |
| Post-perception masked | 379 s | 405 s | 1.23 GB |
| Strict pre-perception masked | 451 s | 475 s | 1.23 GB |

At the current curriculum, only about 7% of all 36 model positions are masked
in expectation. Dynamic gather/scatter overhead exceeds the saved perception
work, so strict execution is **not** a training-speed improvement under this
configuration. Its value here is semantic: masked images cannot enter
perception. No custom recurrent parallel scan is part of this result.

## Gate verdict

- Canonical visible ratio below 1.0 in both seeds: **PASS**
- Every strict masked horizon improves over visible-only: **PASS**
- Every strict masked ratio below 1.0: **PASS**
- Strict within +0.05 of post-perception masked quality: **PASS**
- Current-action probe 4/4 and finite training: **PASS**
- Exact correct-versus-shifted action timing: **UNRESOLVED / non-blocking**
- Training speedup from pre-perception skip: **NOT SUPPORTED**

The functional strict observational-dropout gate passes. The result supports a
useful recurrent state through a 12-step blind target interval after 20 burn-in
steps plus 4 visible target steps. It does not yet prove useful imagined
control or a training-time efficiency advantage.

## Artifacts

- `control_evaluation/`: canonical visible frozen-checkpoint evaluations.
- `evaluation_corrected/`: corrected masked factual evaluations for all six
  checkpoints, with checkpoint/config/split provenance and exact counts.

The superseded first-pass masked JSON files were removed to prevent accidental
reuse.
