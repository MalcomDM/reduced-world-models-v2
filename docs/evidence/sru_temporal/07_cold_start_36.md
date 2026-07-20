# Cold-Start-36 — Stage S4C.0 Two-Seed Experiment

**Date:** 2026-07-19
**Config:** `minimal_sru`, `burn_in=0`, `sequence_len=36`, `random_burn_in`,
10 epochs, batch 8, beta=0.1, K=8 learned, linear head, tokenizer `mean`,
no dropout, cache enabled, 256 held-out windows.

---

## Paired tail-16 comparison

Both checkpoints in each seed are evaluated on the exact same 256 source
segments, with the same shared reward mean and baseline. Each model processes
positions 0--35 from a zero recurrent state and is scored only at positions
20--35. This paired result supersedes the earlier independently sampled SRU-20
values.

### Final state (checkpoint_latest after 10 epochs)

| Metric | CS-36 seed 42 | SRU-20 seed 42 | CS-36 seed 43 | SRU-20 seed 43 |
|--------|:-------------:|:--------------:|:-------------:|:--------------:|
| tail_16 ratio | **1.001** | **0.933** | **1.281** | **0.920** |
| tail_16 MSE | 0.496 | 0.463 | 0.699 | 0.502 |
| Shared baseline MSE | 0.496 | 0.496 | 0.545 | 0.545 |
| Train time | 343 s | 374 s | 373 s | 399 s |
| Peak GPU | 1.23 GB | 1.23 GB | 1.23 GB | 1.23 GB |
| Optimizer updates | 4,650 | 5,050 | 4,980 | 5,380 |
| Model positions | 1,339,200 | 1,454,400 | 1,434,240 | 1,549,440 |
| Supervised targets | 1,339,200 | 646,400 | 1,434,240 | 688,640 |

### Best checkpoints

| Metric | CS-36 seed 42 | SRU-20 seed 42 | CS-36 seed 43 | SRU-20 seed 43 |
|--------|:-------------:|:--------------:|:-------------:|:--------------:|
| Paired tail_16 MSE | 0.443 | 0.436 | 0.491 | 0.394 |
| Paired tail_16 ratio | **0.893** | **0.879** | **0.901** | **0.722** |

Cold-start best checkpoints were selected using all-36 validation, whereas
SRU-20 best checkpoints were selected using target-only validation. The
latest-checkpoint row is therefore the strict fixed-epoch comparison.

Cold-start's own best all-36 ratios were 0.903 (seed 42) and 0.906 (seed
43), with 4/4 action probes in both seeds.

## Answers

**1. Does full 36-step supervision remove the need for separate burn-in?**
No under this random-window training protocol. Cold-start-36's paired
tail_16 ratios (1.001, 1.281) are worse than SRU-20's (0.933, 0.920) at
the same 10-epoch budget. Repeated direct supervision during artificial
cold starts degrades the mature-state predictor.

**2. Does cold-start quality recover by positions 20–35?**
Partially. Seed 42 is near baseline at the final checkpoint and its best
checkpoint is close to SRU-20. Seed 43 remains substantially worse under
both best and final comparisons. Recovery is not robust across splits.

**3. Is it more training-efficient at the same 10-epoch budget?**
No. CS-36 uses fewer optimizer updates (4,650 vs 5,050) but processes
2× the supervised targets with worse held-out quality. Per-update
efficiency is lower.

**4. Is any difference consistent across both seeds?**
SRU-20 consistently beats CS-36 on tail_16 in both seeds. The cold-start
transient is a measurable handicap that additional supervised targets
(at 2× the count) do not compensate for.
