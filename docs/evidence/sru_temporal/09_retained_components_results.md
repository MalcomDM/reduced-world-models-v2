# S4C — Retained-Component Confirmation

**Status:** complete after canonical re-evaluation  
**Date:** 2026-07-19

## Evaluation correction

The first report was invalid and was removed. Its custom evaluator estimated
the reward baseline from only two training files and 50 windows, selected
validation windows with a different RNG convention, executed masked frames
with `post_perception`, labelled raw tokenizer SSE as MSE, and reversed the
lower-is-better ratio comparison.

The results below use only the official evaluators:

- `scripts/evaluation/evaluate_checkpoint.py`;
- `scripts/evaluation/evaluate_masked_dynamics.py`;
- the complete 16-file training split for the constant-reward baseline;
- 256 fixed held-out windows and 4,096 visible target transitions;
- target-relative masks anchored to `loss_mask`;
- tokenizer posterior mean and strict `pre_perception_skip`;
- `checkpoint_best.pt` for every gate.

The four newly trained checkpoints were valid and were not retrained.

## Matched training resources

Every run used MinimalSRU, 20 burn-in positions, 16 supervised target
positions, random-burn-in training, strict observational dropout, beta 0.1,
batch size 8, 10 epochs, the frame cache, and the same seed-specific data
split. Only selection mode or K changed.

| Variant | Seed | Best val MSE | Train time | Peak GPU | Updates | Model positions | Supervised targets |
|---|---:|---:|---:|---:|---:|---:|---:|
| Learned K=8 | 42 | 0.5000 | 451.0 s | 1.23 GB | 5,050 | 1,454,400 | 646,400 |
| Learned K=8 | 43 | 0.5122 | 474.9 s | 1.23 GB | 5,380 | 1,549,440 | 688,640 |
| Fixed-random K=8 | 42 | 0.4477 | 449.4 s | 1.23 GB | 5,050 | 1,454,400 | 646,400 |
| Fixed-random K=8 | 43 | 0.4499 | 476.4 s | 1.23 GB | 5,380 | 1,549,440 | 688,640 |
| Learned K=16 | 42 | 0.4893 | 448.4 s | 1.23 GB | 5,050 | 1,454,400 | 646,400 |
| Learned K=16 | 43 | 0.4644 | 471.6 s | 1.23 GB | 5,380 | 1,549,440 | 688,640 |

Mean training time is 463.0 s for learned K=8, 462.9 s for fixed-random K=8,
and 460.0 s for learned K=16. The four new runs cost 1,845.7 s (30.8 minutes).
All six checkpoints represent 2,771.6 s (46.2 minutes), including the two
pre-existing K=8 anchors.

K does not materially change current runtime or memory: all 225 patches still
pass through the encoder, tokenizer, and scorer before selection, and the
temporal backend receives one pooled spatial representation.

## Canonical visible reward prediction

Ratio is model MSE divided by the constant-mean baseline MSE; lower is better.

| Variant | Seed | Model MSE | Baseline MSE | Visible ratio | Action probe |
|---|---:|---:|---:|---:|---:|
| Learned K=8 | 42 | 0.5000 | 0.5634 | 0.8875 | 4/4 |
| Learned K=8 | 43 | 0.5122 | 0.6295 | 0.8138 | 4/4 |
| Fixed-random K=8 | 42 | 0.4477 | 0.5634 | 0.7946 | 4/4 |
| Fixed-random K=8 | 43 | 0.4499 | 0.6295 | 0.7148 | 4/4 |
| Learned K=16 | 42 | 0.4893 | 0.5634 | 0.8686 | 4/4 |
| Learned K=16 | 43 | 0.4644 | 0.6295 | 0.7377 | 4/4 |

## Strict masked factual reward prediction

Each cell uses correct previous-action history. The per-seed mean averages the
five H ratios; the cross-seed mean then averages the two seed means.

| Variant | Seed | H=1 | H=2 | H=4 | H=8 | H=12 | Seed mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Learned K=8 | 42 | 0.915 | 0.928 | 0.907 | 0.918 | 0.935 | 0.920 |
| Learned K=8 | 43 | 0.965 | 0.951 | 0.947 | 0.950 | 0.955 | 0.953 |
| Fixed-random K=8 | 42 | 0.941 | 0.995 | 0.954 | 0.952 | 0.963 | 0.961 |
| Fixed-random K=8 | 43 | 0.945 | 0.991 | 0.976 | 0.970 | 0.967 | 0.970 |
| Learned K=16 | 42 | 0.917 | 0.928 | 0.890 | 0.898 | 0.916 | 0.910 |
| Learned K=16 | 43 | 1.025 | 0.981 | 0.961 | 0.959 | 0.960 | 0.977 |

The evaluator scored exactly `256 × H` transitions at every horizon. Runtime
metadata confirms `loss_mask` anchoring, tokenizer mean, recurrent burn-in 20,
and `pre_perception_skip`.

## Cross-seed decision table

| Variant | Mean visible ratio | Mean masked ratio | Mean train time | Peak GPU |
|---|---:|---:|---:|---:|
| Learned K=8 | 0.8506 | **0.9370** | 463.0 s | 1.23 GB |
| Fixed-random K=8 | **0.7547** | 0.9656 | 462.9 s | 1.23 GB |
| Learned K=16 | 0.8031 | 0.9434 | 460.0 s | 1.23 GB |

Fixed-random selection predicts visible rewards better, but learned selection
is better through blind intervals by 0.0286 ratio points. Zeroing previous
actions worsens the cross-seed masked mean for every variant; correct versus
one-step-shifted actions remains close and is not treated as an action-index
identification result.

## Tokenizer inference policy

No retraining was performed. Mean uses the learned-K=8 checkpoints.

| Seed | Mean ratio | Sample ratios (RNG 42/43/44) | Sample mean |
|---|---:|---|---:|
| 42 | 0.8875 | 0.8911 / 0.8860 / 0.8871 | 0.8881 |
| 43 | 0.8138 | 0.8185 / 0.8129 / 0.8182 | 0.8165 |
| Cross-seed | **0.8506** | — | 0.8523 |

Posterior mean is deterministic at the model-policy level. Repeated CUDA
pipeline evaluation was exact for seed 43 and differed by only
`9.9e-9` ratio for seed 42, attributable to floating-point kernel reduction
rather than tokenizer sampling. Sampling adds visible variation and is
slightly worse on average.

## Gates

| Decision | Gate result | Evidence |
|---|---|---|
| Learned selection | **PASS — retain learned K=8** | Masked mean 0.9370 beats fixed-random 0.9656 by 0.0286. Fixed-random's visible advantage is recorded as a real trade-off, not ignored. |
| K=16 | **PASS — do not adopt** | K=16 masked mean 0.9434 is 0.0065 worse than K=8, far from the required improvement greater than 0.02; it admits twice as many selected patches. |
| Tokenizer mean | **PASS with CUDA numerical qualification — retain mean** | Mean cross-seed ratio 0.8506 beats sampled mean 0.8523. Sampling has no consistent/material advantage; numerical mean-policy drift is below 1e-8. |
| Action sensitivity | **PASS** | Every best checkpoint produces 4/4 distinct current-action-conditioned predictions. |

## Decision

The corrected evidence preserves the previously selected defaults:

- learned Top-K;
- K=8;
- tokenizer posterior mean at evaluation.

The result also exposes a useful distinction: static random coverage is better
for fully visible reward regression, while learned selection is more robust
when the recurrent model must continue through missing observations. S5 should
therefore keep learned K=8 and test whether that masked robustness transfers
to imagined control.
