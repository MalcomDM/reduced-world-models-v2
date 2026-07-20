# SRU Temporal Lineage — Matched Validation Plan

**Status: COMPLETE — MinimalSRU adopted on `main` (2026-07-20).**

## Purpose

Replace the bounded-history causal Transformer with a compact SRU-like
recurrent state **only if** it reaches the already achieved causal-transformer
checkpoint with comparable or better quality, cost, and stability.

This is a focused temporal-backend comparison, not a restart of the complete
project. The perception stack, K=8 learned selection, stochastic tokenizer,
linear reward head, transition timing, datasets, cache, Actor-Critic losses,
and locked evaluation seeds remain fixed.

Historical baseline: `runs/component_refinement/causal_transformer/`.
New artifacts: `runs/component_refinement/sru_temporal/`.

Frozen code baseline:

- branch: `baseline/causal-transformer-stage5`
- commit: `a0406262c6b3b2db647a7675e4bd6125e157b16f`
- verification: `pytest` — 416 passed

## Development Strategy

- The verified causal-transformer state is preserved on
  `baseline/causal-transformer-stage5`.
- Continue development on `main`.
- Temporarily support `causal_transformer` and `sru` behind one temporal-model
  interface so matched experiments use the same data, trainer, losses, and
  evaluator.
- Keep the downstream state width at 80 when possible. SRU-only dimensions
  stay inside the temporal module.
- Remove the causal backend only after the final decision gate; the baseline
  branch remains as the reproducible historical implementation.

## Fixed Temporal Contract

At decision time, the SRU hypothesis must satisfy:

```text
x_t = concat(mask_t * spatial_t, previous_action_t, mask_t)
z_t = SRU(x_t, z_{t-1})
reward = R(z_t, current_action_t)
actor/value = Actor/Critic(ControllerTrunk.encode(z_t))
```

- `z_t` is the complete recurrent temporal state needed for the next step.
- Actions remain visible when observations are masked.
- No explicit next-image or next-latent prediction loss is added.
- The exact SRU equations are frozen before implementation. Parallel input
  projections are expected, but recurrent-scan parallelism is measured rather
  than assumed; incremental inference carries only `z_t`.

## Performance invariants and remaining risks

The causal-line performance work remains part of the SRU implementation:

- Frame tensors use the explicit, hash-validated memory-mapped cache; cache
  use is recorded in every experiment config.
- Perception and reward-head work remain vectorised over flattened ``B*T``.
- The SRU input projection is one fused ``Linear`` over ``B*T``; gradients from
  every step accumulate into the same parameters normally.
- Loaders retain pinned memory, persistent workers, and the measured six-worker
  default on the main random-window training path.
- Incremental inference carries only ``z_t`` and performs one SRU cell step;
  it never rebuilds the causal 20-token history.
- Masked frames can bypass perception through ``pre_perception_skip``.
  Its dynamic gather/scatter overhead means speed must be measured from the
  actual visible-frame fraction rather than inferred from mask probability.

The recurrence over ``T`` remains an eager sequential scan. Candidate/carry
projections are parallel over ``B*T``; only the cheap elementwise state update
is sequential. A custom associative/CUDA scan was considered and **discarded
for this thesis line**: the available prototype lacks autograd, a production
version would require custom forward/backward kernels, and the estimated
full-pipeline gain is too small relative to that implementation and validation
risk. Reconsider it only if a later matched profile identifies the recurrence
itself as a material blocker. Before S6 selects a backend, run a matched CUDA
full-pipeline benchmark and report perception, recurrent scan, backward,
loader, memory, and incremental latency separately. Do not select SRU based
only on its parameter count or temporal-only CPU benchmark.

## Checkpoints

### S0 — Freeze baseline and backend contract

- Commit the current causal implementation, create the baseline branch, and
  record its commit in the SRU result index.
- Define the common sequence, single-step, state, output, config, and checkpoint
  interfaces.
- Define the exact SRU equations and the shuffled-window state-initialisation
  rule. Arbitrary windows must use episode-prefix processing, a loss-free
  same-episode burn-in, or a version-valid state cache; zero reset and
  `predecessor_action` alone are not sufficient.
- Prove old causal checkpoints still load unchanged.

**Gate:** both backends can coexist without changing causal outputs or existing
training behavior. No recurrent state crosses an episode boundary, and reward
metrics are reported by distance from state reset.

### S1 — SRU mechanics

- Implement the smallest SRU-like temporal module and its recurrent state.
- Test sequence/incremental parity, action timing, state reset, gradients,
  deterministic evaluation, masking, checkpoint round-trip, and `z_t`-only
  resume.
- Test the selected burn-in/state-initialisation path and prove that its loss
  mask excludes burn-in positions.
- Benchmark parameters, MACs, training throughput, incremental latency, blind
  rollout latency, memory, and GPU use against the causal backend.

**Gate:** exact interface semantics, no history bypass, finite gradients, and
measured efficiency evidence. Review implementation before long training.

### S2 — Visible reward anchor

- Run the existing nonconstant-window overfit diagnostic.
- Train matched K=8, beta=0.1, linear-head, mean-tokenizer anchors for seeds 42
  and 43 with the same dataset split, cache, 10 epochs, and validation cap.
- Record held-out MSE/MAE, baseline ratio, action probe, epoch time, throughput,
  and peak memory.

**Gate:** both seeds beat the constant-mean baseline. Flag an absolute
cross-seed mean-ratio regression greater than 0.05 against the causal visible
reference for investigation before continuing.

### S2.5 — Random macroblock burn-in + truncated-BPTT ablation

The S2 random-window anchor reconstructs 20 state-building frames for every
16 target frames. The selected efficiency protocol instead samples independent
same-episode **macroblocks** in random order:

```text
[20-step burn-in, no direct loss] → [64 target steps]
```

- Partition every episode into non-overlapping 64-step target regions; the
  first target block starts from ``z_0 = 0`` and later blocks load their own
  preceding 20-step burn-in. Target regions never overlap, and the final
  partial loader batch is retained so one macro pass covers every target once.
- Randomise macroblock order and batch macroblocks independently. No recurrent
  state or cache crosses macroblock boundaries.
- Within a macroblock, process the four 16-step target chunks progressively,
  carry ``z_t`` between chunks, and detach it after every optimizer update.
- Direct reward MSE and tokenizer KL apply only to target positions. The first
  target update may backpropagate through its real burn-in prefix.
- The initial 64-step candidate limits stale-state exposure to four updates
  while reducing repeated context from ``20 / 16`` to ``20 / 64`` target
  steps. A later ``64 vs 96`` ablation may trade more state freshness for less
  repeated context, while retaining random batches and avoiding stale cached
  states.

``sru_training_mode="random_burn_in"`` remains the S2 baseline/default.
The already implemented ``sequential_tbptt`` mode is retained as an optional
diagnostic but is not the selected S2.5 training protocol.

**Gate:** verify exact macroblock boundaries, burn-in/action timing,
within-block state hand-off/detach, and target-only losses. Then run matched
seed-42/43 visible anchors and compare quality, wall time, optimizer updates,
real transition exposures, and processed model positions against S2.

For the first comparison, train the full world model only (perception,
tokenizer, spatial selector, MinimalSRU, and linear reward head). Actor and
critic are absent; observational masking is disabled. Report learning curves
against direct supervised target exposure, processed model positions, optimizer
updates, and elapsed time. A macroblock refreshes its burn-in state with current
weights; only the three later target chunks use a state produced before their
immediately preceding optimizer step. This result therefore does not establish
end-to-end actor--critic stability or blind-imagination quality.

### S2.5C — Superseded macroblock burn-in ablation

The M=64 macroblock experiment did not preserve visible reward quality against
SRU random burn-in. Do not run its burn-in-length sweep as a selection study.
Its artifact is retained as an exploratory efficiency result only; the
canonical burn-in study is defined in S4B.

### S3 — True observational dropout

- Train the matched temporal-mask curriculum with the observation removed from
  the recurrent input while `z_{t-1}` continues the dynamics.
- Evaluate visible reward quality; blind horizons 1, 2, 4, 8, and 12; correct,
  zero, and shifted action histories; and visible-state resynchronisation.
- Prove a saved `z_t` resumes identically without observation or temporal
  history.

**Current evidence:** post-perception masking supports blind reward prediction
at H=1/2/4/8 relative to matched SRU visible-only anchors. The strict
pre-perception execution and the H=12 matched control remain pending in S4A.
Seed-43 action-history sensitivity is marginal at short horizons, so it is not
yet a strict action-conditioning conclusion.

### S4A — Strict pre-perception observational-dropout mechanics

The current mask zeroes the spatial representation *after* CNN/tokenizer/
selection work. It does not leak visual data into the temporal state, but it
still computes posterior statistics and KL gradients for masked images. Add an
explicit ``pre_perception_skip`` execution policy:

- visible positions use the unchanged perception path;
- masked positions bypass CNN, tokenizer, scorer, selector, and spatial head;
- masked positions provide zero spatial features and ``keep_bit=0``;
- tokenizer KL is reduced only over visible supervised positions;
- all-visible execution is exactly compatible with the existing path.

**Mechanics gate:** all-visible output/state parity; masked reward/state parity
against post-perception masking in evaluation; no perception invocation or
perception gradient for masked frames; finite gradients for visible frames;
and a measured frame-count/time/memory benchmark. Existing post-perception
dropout anchors remain functional evidence, not strict-dropout baselines.

### S4B — Canonical visible temporal and burn-in comparison

Create a new, isolated comparison family. Every run uses the same current code,
source corpus/cache, seeds 42/43, K=8, beta=0.1, linear reward head,
posterior-mean evaluation, batch size 8, 10 ordinary dataset epochs, 256 held-
out windows, and the same frozen-checkpoint evaluator. Observational dropout is
disabled for this study.

| Variant | Purpose |
|---|---|
| Causal Transformer | canonical temporal reference |
| SRU random burn-in 20 | recurrent quality reference |
| SRU random burn-in 8 | reduced-context candidate |
| SRU random burn-in 4 | minimal-context candidate |
| SRU random burn-in 0 | intentional no-context lower bound |

Report per seed and mean: best/final and frozen-evaluator MSE/ratio, direct
targets, model positions, updates, separate train/validation time, GPU memory,
and action probe. Keep macroblock outside this family.

**Gate:** choose the smallest recurrent burn-in that beats the constant
baseline in both seeds and remains within a predeclared mean-ratio tolerance of
SRU-20. The tolerance must be recorded before runs begin (recommended initial
value: 0.02 ratio points). Architecture selection remains deferred to S6.

**Result (complete):** the corrected frozen-checkpoint protocol selected
SRU-20. Mean ratios were Causal 0.825, SRU-20 0.736, SRU-8 0.843, SRU-4 0.842,
and SRU-0 0.849. None of the reduced contexts met the 0.02 tolerance from
SRU-20. The apparent SRU-20 advantage has substantial two-seed spread, so this
is a context decision only; backend selection remains deferred to S6.

### S4B.1 — Strict observational-dropout anchor at the selected context

After S4B selects the recurrent burn-in, train two fresh masked-reward anchors
(seeds 42/43) with that burn-in and ``pre_perception_skip``. Compare them with
the corresponding post-perception masked control using the same visible and
masked factual protocol. Report the actual visible-frame fraction and separate
perception/train timing; do not attribute a speedup merely from the configured
mask probability.

**Gate:** both strict anchors remain below the constant baseline when visible,
improve every matched blind-horizon score over their visible-only control, and
have finite loss/action probes. This is the strict observational-dropout
evidence used by later imagination work.

**Corrected result (complete):** the first evaluation incorrectly started
``warmup=4`` at layout position zero. The frozen checkpoints were re-evaluated
with burn-in always visible and mask/scoring anchored to ``loss_mask``; no
retraining was performed. Strict mean masked ratios are 0.920 (seed 42) and
0.953 (seed 43), versus visible-only 0.954 and 1.023. Every strict H=1/2/4/8/12
ratio improves over the matched visible-only control and remains below 1.0.
Strict stays within +0.004/+0.019 of post-perception mean masked quality.
Correct-versus-shifted action timing remains marginal, and strict execution is
17--19% slower in training at the current approximately 7% masked-position
rate; neither point blocks the functional observational-dropout gate.

### S4C.0 — Cold-start 36-step supervision probe

**Two-seed experiment complete.** Compare the canonical
SRU layout (20 context positions plus 16 supervised targets) with an SRU
cold-start layout containing 36 directly supervised positions and no separate
burn-in. Both process a maximum recurrent path of 36 positions. The cold-start
checkpoint is evaluated both over all 36 positions and over only positions
20--35; the latter is the directly comparable metric against canonical SRU-20.
This is a practical training-policy comparison, not a matched-target-exposure
or single-cause ablation. On paired source windows, SRU-20 beats cold-start-36
at the final checkpoint in both seeds (ratios 0.933 versus 1.001, and 0.920
versus 1.281). Directly supervising artificial cold-start positions does not
replace loss-masked state reconstruction; retain the 20-step context protocol.

### S4C — Minimal retained-component confirmation

- Recheck learned K=8 against a static selection control.
- Compare K=8 with K=16 only.
- Compare mean versus sampled tokenizer inference without retraining.

Do not repeat already settled KL-reduction, nonlinear-head, cache, or generic
perception experiments unless the SRU results expose a regression.

**Gate:** the retained defaults remain defensible, or the changed conclusion is
recorded explicitly.

**Corrected result (complete):** four matched strict-SRU anchors were trained
for fixed-random K=8 and learned K=16, while the existing learned-K=8 anchors
served as the baseline. Official full-split evaluation gives cross-seed masked
ratios 0.937 (learned K=8), 0.966 (fixed-random K=8), and 0.943 (learned K=16).
Fixed-random is stronger on fully visible reward regression, but learned K=8
is more robust through blind intervals. K=16 does not meet the required
greater-than-0.02 improvement, and sampled tokenizer inference is slightly
worse on average than posterior mean. Retain learned K=8 and mean inference
for S5. The superseded custom evaluation was removed because it used a partial
baseline and reversed the lower-is-better gate.

### S5 — Imagination and frozen Actor-Critic parity

- Replace history-based imagination with `z_t`-only score-then-advance.
- Re-run imagination mechanics, frozen-world hash, critic-learning, entropy,
  and action-bound gates.
- Match the causal Stage-5 budget: seed 42, 2,000 updates, horizons 1/2/4,
  entropy 0.03, then evaluate locked dev seeds 100–102 for 1,000 steps.
- Include the same zero-action control and a formally recorded random-action
  control.

**Gate:** finite stable training, decreasing critic loss, unchanged frozen
world model, and real return that matches or exceeds the causal Actor reference
within a predeclared tolerance. A second Actor-Critic seed is required before
deleting the causal backend, not before the first matched checkpoint.

### S6 — Architecture decision (complete)

**Decision:** adopt `MinimalSRUTemporal` as the primary temporal backend on
`main`. Preserve the reproducible causal implementation on branch
`baseline/causal-transformer-stage5`; do not use it as the default for new
experiments.

| Decision evidence | Causal Transformer | MinimalSRU | Interpretation |
|---|---:|---:|---|
| Temporal parameters | 56,560 | 5,920 | SRU is 9.6× smaller. |
| Incremental CPU latency | 0.47 ms | 0.051 ms | SRU is about 9.2× faster for one temporal step. |
| Matched visible reward ratio, seeds 42/43 | 0.865 / 0.779 | 0.839 / 0.761 | SRU preserves factual reward quality. |
| Strict blind reward prediction | Bounded-history masking | H=1–12 ratios below 1.0 | SRU carries a genuine recurrent `z_t`. |
| Frozen Actor real return | −75.3 | −82.3 | Causal is 7.0 points better; SRU passes the predeclared −85.3 parity gate. |
| Resume state | history + lengths | `z_t` only | SRU enables compact latent starts and cheap blind rollout. |

This is an architecture-viability decision, not a solved-control claim.
Random actions still outperform both frozen imagined policies on the current
three-seed diagnostic. The remaining limitation is policy/state sensitivity,
not a mechanical failure of the SRU transition.

## Return to the Main Plan

This validation line ends at parity with the current **Stage 5.3** checkpoint:
a frozen world model, trained Actor-Critic, and locked real-environment
evaluation. It does not implement joint end-to-end behavior gradients.

After S6, resume the main plan at **Stage 6 — progressive joint behavior
gradients**, using MinimalSRU as the selected backend.

Stages 0–5 are not repeated after this decision. Stage 7 collection/replay and
the optional memory experiments remain downstream of the Stage-6 safety gates.

## Causal Reference Values

| Evidence | Causal reference |
|---|---:|
| Visible-only reward ratio, seeds 42 / 43 | 0.845 / 0.767 |
| Masked-trained visible ratio, seeds 42 / 43 | 0.882 / 0.853 |
| Masked-trained blind ratio range, seeds 42 / 43 | 0.928–0.954 / 0.865–0.925 |
| Temporal / complete world-model parameters | 56,560 / 93,524 |
| Temporal incremental cost at T=20 | about 1.14M MACs and 0.93 ms |
| Frozen Actor real return, locked seeds 100–102 | −75.3 mean |
| Zero-action baseline, locked seeds 100–102 | −92.9 mean |

These are comparison anchors, not success thresholds for solved driving.

## S5.0 — Frozen SRU Actor-Critic (this experiment)

| Evidence | SRU value |
|---|---|
| Frozen Actor real return, locked seeds 100–102 | **−82.3** mean (seeds 100: -81.5, 101: -83.5, 102: -82.1) |
| Zero-action baseline (same seeds) | −92.9 mean |
| Deterministic random baseline (same seeds) | −28.2 mean |
| Anchor verified | ✓ (hash 79326840535458c5) |
| Checkpoints | 500, 1000, 1500, 2000 |
| Total imagined transitions | 37,232 |
| Critic loss (first→last 100 median) | 0.0495 → 0.0159 |
| World model / ControllerTrunk frozen | ✓ |
| Target Critic no gradients, not in optimizer | ✓ |

**Primary parity gate: SRU mean real return ≥ -85.3 → PASS** (−82.3 ≥ −85.3)

Full report: `../evidence/sru_temporal/10_frozen_actor_critic_parity.md`.
Raw checkpoints, metrics, JSON, CSV, and plots remain under
`runs/imagined_actor_critic/minimal_sru/01_frozen_parity/seed42/`.

## Artifact Layout

```text
docs/evidence/sru_temporal/
├── 01_visible_reward_anchor.md
├── ...
├── 10_frozen_actor_critic_parity.md
└── run_index.md

runs/component_refinement/sru_temporal/
├── 01_visible_reward_anchor/
├── 02_macroblock_m64_matched_exposure/
├── 03_matched_backend_evaluation/
├── 04_observational_dropout_anchor/
├── 05_canonical_burnin_comparison/
├── 06_sru4_matched_wallclock/
├── 07_cold_start_36/
├── 08_strict_observational_dropout_anchor/
├── 09_retained_components/
└── RUN_INDEX.md

runs/imagined_actor_critic/minimal_sru/
└── 01_frozen_parity/seed42/
    ├── checkpoints/            (ac_checkpoint_{500,1000,1500,2000}.pt)
    ├── evaluation/
    │   ├── actor/              (per-seed CSV, JSON, summary, plots)
    │   ├── zero/               (per-seed CSV, JSON, summary)
    │   └── random/             (per-seed CSV, JSON, summary)
    ├── metrics.csv
    ├── training_summary.json
    ├── fixed_probe_pre.json
    ├── fixed_probe_post.json
    └── RESULTS.md
```

Tracked reports live under `docs/evidence/sru_temporal/`. Raw generated
checkpoints, JSON, CSV, plots, and environment manifests remain under ignored
`runs/` directories. Every report records the source lineage, config,
dataset/seed protocol, metrics, and explicit conclusion/non-conclusion.
