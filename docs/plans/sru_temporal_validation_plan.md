# SRU Temporal Lineage — Matched Validation Plan

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

### S3 — True observational dropout

- Train the matched temporal-mask curriculum with the observation removed from
  the recurrent input while `z_{t-1}` continues the dynamics.
- Evaluate visible reward quality; blind horizons 1, 2, 4, 8, and 12; correct,
  zero, and shifted action histories; and visible-state resynchronisation.
- Prove a saved `z_t` resumes identically without observation or temporal
  history.

**Gate:** visible ratio remains below 1.0; masked training improves every
matched blind-horizon evaluation over the SRU visible-only anchor; correct
actions outperform the controls; degradation is bounded and finite.

### S4 — Minimal retained-component confirmation

- Recheck learned K=8 against a static selection control.
- Compare K=8 with K=16 only.
- Compare mean versus sampled tokenizer inference without retraining.

Do not repeat already settled KL-reduction, nonlinear-head, cache, or generic
perception experiments unless the SRU results expose a regression.

**Gate:** the retained defaults remain defensible, or the changed conclusion is
recorded explicitly.

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

### S6 — Architecture decision

Compare in one table:

- visible and masked reward quality;
- training time, throughput, latency, memory, parameters, and MACs;
- blind-horizon and action-history behavior;
- Critic learning, policy stability, and locked real-environment return;
- implementation complexity and recurrent-state semantics.

Choose one outcome: adopt SRU and remove causal code from `main`; retain both
because evidence is mixed; or block SRU and continue from the causal baseline.

## Return to the Main Plan

This validation line ends at parity with the current **Stage 5.3** checkpoint:
a frozen world model, trained Actor-Critic, and locked real-environment
evaluation. It does not implement joint end-to-end behavior gradients.

After S6:

- if SRU is adopted, make it the selected temporal backend and resume the main
  plan at **Stage 6 — progressive joint behavior gradients**;
- if evidence is mixed, select and record one backend before entering Stage 6;
- if SRU is blocked, restore the causal baseline and resume Stage 6 from its
  existing Stage-5.3 checkpoint.

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

## Artifact Layout

```text
runs/component_refinement/sru_temporal/
├── 00_mechanics/
├── 01_visible_reward_anchor/
├── 02_observational_dropout_anchor/
├── 03_component_confirmation/
├── 04_imagination_actor_critic/
└── RESULTS.md
```

Every result records the source commit, causal baseline commit, config,
dataset/seed manifest, command, metrics, and explicit conclusion/non-conclusion.
