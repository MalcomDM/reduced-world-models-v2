# Architecture Hypothesis and Validation Plan

## Purpose

This document records the architecture audit performed after the first usable
end-to-end reward predictor. It separates:

1. structural correctness;
2. evidence that a mechanism is actually used;
3. evidence that the mechanism improves prediction or control.

The Stage 2 checkpoint is the anchor for the experiments below. The project
should not enable every unfinished mechanism simultaneously: correctness fixes
may be grouped, but each research hypothesis must be compared against a matched
end-to-end baseline.

## Refined Research Hypothesis

> A shared variational patch bottleneck, learned sparse spatial selection, and
> masked action-conditioned temporal context can learn a compact,
> task-sufficient representation for visual continuous control when shaped by
> factual reward, value, and policy objectives, without pixel reconstruction or
> an explicit next-state prediction loss.

The intended representation is allowed to be incomplete or physically
inaccurate if it preserves what the policy needs. The thesis does **not** assume
that it will recover true geometric factors, disentangled variables, or
human-interpretable components.

Recommended terms are **task-sufficient**, **reward/value-predictive**,
**variational patch bottleneck**, **sparse selection**, and **emergent
dynamics**. “Same token,” “semantic component,” and “human-like” require direct
measurement and must not be used as conclusions from task performance alone.

## Theory Clarifications

### Variational patch tokens

The tokenizer is genuinely stochastic during training: a shared patch mapping
produces `mu` and `logvar`, and samples `z = mu + sigma * epsilon`. Evaluation
uses `mu`.

The defensible hypothesis is narrower than “sampling makes similar inputs the
same”:

- the shared CNN and patch projection are the first reasons similar local
  inputs may obtain nearby representations;
- sampling injects noise and forces downstream computations to tolerate a
  neighbourhood of each encoding;
- a correctly weighted KL term to a shared prior creates an information/rate
  bottleneck; combined with task losses, task-interchangeable inputs may reuse
  overlapping posterior regions;
- sampling alone does not pull two inputs together, and with `beta=0` the model
  can simply reduce posterior variance;
- neither KL nor continuity guarantees semantic clusters or disentanglement.

Token similarity must be measured on posterior means **before positional
encoding**. Position is intentionally added afterward so equal content at two
locations remains distinguishable to the selector.

The present `beta=0` reward anchor is not evidence for or against this
hypothesis: its KL statistic is logged but does not affect optimisation. After
the per-token KL reduction is corrected, train matched `beta=0` and small
nonzero-beta variants, then compare posterior means on predefined same-region
and cross-region patches across fixed evaluation episodes. PCA token colours
remain an exploratory visual aid only, not a similarity result.

### Observational dropout

Freeman, Metz, and Ha show that useful dynamics can emerge from optimizing real
task return while observations are intermittently hidden, without a supervised
forward-prediction loss. “Without forward prediction” refers to the **loss**:
their model still applies an action-conditioned transition and recursively
carries its generated state until the next real observation resynchronizes it.

The current reward-only Stage 2 experiment deliberately forces observations to
remain visible. It therefore validates the base pipeline, not the
observational-dropout hypothesis.

The current Transformer can compute a bounded projected state from the last
visible context plus an action sequence, but it does not append its generated
belief as the next temporal input. Once the last real perceptual token leaves
the finite context window, no generated state is preserved. This is not proof
that the bounded Transformer approach is invalid, but it must pass masked
horizon tests before being described as an emergent dynamic model.

### Minimal DeepMDP/value-equivalence lesson

DeepMDP gives a useful diagnostic: two states are behaviourally similar when
they produce similar rewards for actions and transition to behaviourally
similar future states. Reward prediction alone cannot guarantee this. For
example, two straight-looking positions can have the same immediate reward but
require opposite actions because different curves follow.

An explicit next-latent loss is **not** required for this thesis. The minimal,
aligned approximation is:

1. train factual reward prediction under full observations;
2. warm up on visible frames, then mask contiguous observation horizons while
   replaying the recorded actions;
3. require useful reward predictions at horizons 1, 2, 4, 8, and 16;
4. once the Critic exists, train full and masked beliefs against the same
   factual lambda-return target;
5. only if needed, add a small stop-gradient consistency loss between
   full-observation and masked **value/policy outputs**, not between raw latent
   vectors.

This is value-equivalence-inspired evidence, not a DeepMDP guarantee. It keeps
the structure small and tests functional equivalence rather than visual or
latent reconstruction.

### Gradient boundary

The thesis may allow factual reward and real/bootstrapped Critic losses to shape
the controller, temporal model, selector, tokenizer, and CNN. A policy-gradient
loss grounded in real environment return may also be tested upstream in stages.

During an Actor update whose objective is computed from **imagined learned
rewards**, reward-model and dynamics parameters must be frozen for that update.
The Actor may differentiate through their operations with respect to its
actions, but it must not improve imagined return by changing the learned judge
or imagined environment. Controlled full coupling remains a later ablation,
not the safe default.

## What Current Evidence Does and Does Not Establish

### Established

- The complete reward path can overfit a nonconstant 16-step window to
  `MSE ~= 0.0004` with `beta=0`; end-to-end gradients are connected.
- On the existing episode-level splits, two checkpoints beat a constant-mean
  reward baseline by approximately 14--16 percent.
- The trained selector is driven mainly by learned content logits rather than
  the explicit sinusoidal positional encoding. The exact decomposition is
  possible because the scorer is linear and bias-free:

  | Checkpoint | Held-out frames | content logit SD | position logit SD | total/content Top-8 overlap | total/position overlap |
  |---|---:|---:|---:|---:|---:|
  | seed 42, beta 0 | 1,059 | 1.057 | 0.238 | 7.22 / 8 | 0.09 / 8 |
  | seed 43, beta 0 | 793 | 1.024 | 0.201 | 7.28 / 8 | 0.09 / 8 |

  The probe computes `Scorer(mu)`, `Scorer(position_encoding)`, and
  `Scorer(mu + position_encoding)` on every unique frame in each checkpoint's
  four held-out episodes. It isolates the explicit positional term; it does not
  prove semantic token clustering, because CNN features may still contain
  implicit location and scene regularities.

- The existing rollouts are not filled with zero actions: among 5,339 steps
  there are zero exact all-zero action vectors and 4,928 distinct action vectors
  after rounding to three decimals. All 20 episodes do, however, begin with the
  same 20-step full-gas push.

### Narrow or unresolved evidence

- Normal and shuffled-current-action reward MSE are almost identical on the
  random-policy validation distribution. This establishes only that the
  **direct current-action reward branch** contributes almost nothing there. It
  does not establish that action history is useless, that informative policies
  would not expose action effects, or that masked multi-step dynamics cannot be
  learned.
- The existing reward head is additive,
  `reward = f(belief) + w_action^T action + bias`. It cannot express that the
  value of braking or steering depends on the current state. A small nonlinear
  joint head is therefore a required matched ablation.
- The manual “centered but stopped” observation is valuable qualitative
  counterevidence to a pure static-road shortcut: prediction falls when the car
  stops despite remaining centered. It still cannot identify whether the model
  used action history, visual motion, or the on-screen speed/HUD because the
  current manual log stores only real and predicted rewards.
- The saved manual CSV has been overwritten since the originally documented
  756-step run. The current file has 1,045 rows over two unequal episodes and
  does not contain observations, actions, or environment seeds. It cannot serve
  as a reproducible thesis artifact.
- Reward is `-0.1` on 95.24 percent of the existing transitions. Startup zoom,
  identical early gas, and high initial tile rewards make episode position a
  strong confound. Report startup and steady-driving results separately.

## Confirmed Structural Gaps

These are implementation facts, not rejected research hypotheses.

### Correctness gates

1. **Disconnected Top-K surrogate (FIXED Stage 2.5B.2).**
   The selector now returns a K-mass STE mask where the soft surrogate sums
   to ``K`` (``K * softmax(logits/temp)``).  During training,
   ``SpatialAttentionHead`` builds normalised pooling weights over ALL patches
   using this mask:

   .. code:: python

       weights_raw = exp(logits/temp) * selection_mask
       weights = weights_raw / weights_raw.sum(dim=-1, keepdim=True)
       h = sum_i weights[i] * W_v(tokens[i])

   In eval mode the head retains the original K-token value projection and
   hard-gather + softmax(K) pooling exactly. In training the STE mask routes
   gradient to unselected scorer logits through the ``K * softmax`` surrogate.

   The soft surrogate uses ``K`` as total mass (not sum-one), so the
   straight-through correction ``(hard - soft).detach() + soft`` preserves
   the correct forward value while providing dense backward gradients.
2. **False temporal reset at sliding-window starts (FIXED Stage 2.5B.3).**
   Every sampled window previously set its first previous action to zero.  Now
   ``RolloutDataset.__getitem__`` returns a ``predecessor_action`` field:
   ``action[offset-1]`` at mid-episode offsets, zeros only at true episode
   start (``offset == 0``).  The trainer's ``_compute_batch_loss`` and
   ``evaluate`` use this value for ``prev_actions[:, 0]``.

   The corrected contract is:

   ```
   prev_actions[:, 0] = predecessor_action
                      = zeros              if offset == 0  (true episode start)
                      = action[offset-1]   if offset > 0   (mid-episode)
   prev_actions[:, t] = actions[:, t-1]    for t > 0
   ```

   No predecessor crosses an episode boundary because each window is contained
   within a single ``.npz`` file. A missing ``predecessor_action`` is rejected
   by training and evaluation rather than silently restoring the old reset.
3. **Incorrect KL aggregation (FIXED Stage 2.5B.1).**
   ``mu`` and ``logvar`` were averaged over time before applying the nonlinear
   KL expression.  Now the full ``(B, T, P, D)`` posterior is preserved through
   ``forward_sequence`` and the elementwise KL formula is applied before any
   reduction:

   .. code:: python

       kl_per_element = 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
       kl = kl_per_element.mean()

   This is the mathematically correct reduction.  ``beta=0`` remains the
   current anchor; ``beta>0`` comparisons are deferred until all structural
   corrections (including this one) are validated and a retrained checkpoint
   exists.
4. **State-independent action effect.** Replace or ablate the linear reward head
   against a one-hidden-layer joint MLP over `[belief, action]`.
5. **Masked temporal path unvalidated.** Observational dropout is intentionally
   bypassed in Stage 2. Add controlled contiguous masks and correct generated
   state/history semantics before using the imagination code.
6. **Draft imagination timing.** The current simulator advances the belief and
   then scores the old action with the new belief, and duplicates the last real
   spatial representation on the first imagined step. Treat it as inactive
   draft code until the masked temporal contract passes.
7. **Finite positional/context mismatch.** Training uses length 16 while the
   learned temporal table has length 20; positions 16--19 are not trained for
   the validated run. Match training and deployment context or use a
   non-learned/relative temporal position scheme.
8. **Rollout provenance.** Store environment seed, policy identity/version,
   separate `terminated` and `truncated`, collector settings, and the action
   preceding each sampled window.

### Engineering gates

- The normal training CLI path and configuration objects must be exercised by
  an integration test; architecture dataclasses currently record several
  values that model constructors still read from module-level constants.
- Training, dataset, simulator, and manual evaluation preprocessing must share
  one implementation.
- Full-episode evaluation should count each transition once. Current
  stride-one windows repeatedly weight the same held-out frames.

## Fixed Evaluation Data

Gymnasium CarRacing has a keyboard driver but no built-in competent or perfect
automatic policy. `env.reset(seed=N)` does reproduce the same track, so fixed
scientific scenarios are still practical.

Create three immutable groups:

1. **development tracks** for implementation and tuning;
2. **validation tracks** for architecture selection;
3. **locked test tracks** evaluated only for final thesis results.

Collect the following policy strata on fixed seeds:

- random-smooth and scripted action excitation for broad counterfactual input;
- competent human driving or a disclosed privileged waypoint heuristic using
  simulator track/car state only to generate evaluation trajectories;
- later, snapshots of the learned policy at multiple training stages.

A perfect driver is unnecessary. The important requirement is coverage of
straight driving, stopping/braking, left and right curves, recovery, and
off-road failure.

For the strongest action/dynamics test, save a competent action prefix, reset
the environment with the same seed, replay the prefix to reproduce the same
physical state, and branch into several fixed action sequences. Compare masked
predictions with the real rewards from each branch. This tests conditional
dynamics without waiting for a trained Actor.

Persist observations, actions, rewards, environment seed, episode and step,
termination/truncation, policy provenance, early-push schedule, environment
version, and repository commit. Manual trajectories used repeatedly for model
selection are validation data, not final test data.

### Metrics and baselines

Always include:

- natural-distribution MSE/MAE;
- AUPRC for a tile/reward event (`reward > -0.1`), with AUROC secondary;
- error on neutral and positive-reward steps separately;
- cumulative reward error at horizons 1, 2, 4, 8, and 16;
- startup (`0--49`) and steady-driving (`>=50`) strata;
- HUD visible and bottom-region-masked strata.

Required baselines are constant mean, episode-position only, current-frame only,
no/shuffled action history, and no/shuffled temporal history. Using HUD telemetry
is not “cheating” because it is part of the observation; the crop ablation
records that dependency and prevents a road-geometry claim from being made by
accident.

## Checkpointed Execution Plan

### Checkpoint A — Measurement foundation

- Implement seeded rollout provenance and fixed development/validation/test
  track lists.
- Save competent human/heuristic trajectories and controlled action branches.
- Evaluate unique transitions once and add the metrics/baselines above.

**Gate:** identical seed plus identical actions reproduces the same scenario;
splits and artifacts are immutable and checksummed.

### Checkpoint B — Correct and accelerate the Stage 2 anchor

- Fix previous-action offsets, Top-K gradient routing, KL reduction, temporal
  length mismatch, shared preprocessing, and the normal CLI/config path.
- Add the nonlinear joint reward-head option.
- Batch perception/controller over `(batch * time)`, cache resized rollout
  frames, reduce duplicate stride-one work, and lower gradient-log frequency.
- Retrain the corrected reward-only anchor before activating dropout.

**Gate:** all regression tests pass; the corrected anchor at least matches the
current fixed-suite reward result; deterministic evaluation is reproducible;
throughput and memory are recorded.

### Checkpoint C — Perception hypotheses

Retrain matched end-to-end variants, changing one factor at a time:

- learned Top-K versus random Top-K, fixed-position Top-K, and all-token pooling;
- `K = 8, 16, 32`;
- deterministic tokenizer versus stochastic tokenizer with corrected small
  beta values;
- linear versus nonlinear reward head.

Measure task metrics, posterior statistics before positional encoding,
selection heatmaps/diversity, parameters, MACs, latency, and memory.

**Gate:** retain a mechanism as a thesis contribution only if it improves a
declared quality/efficiency trade-off across multiple seeds. A connected
gradient or an attractive heatmap is mechanism evidence, not task evidence.

### Checkpoint D — Emergent masked dynamics

- Warm up on 5--10 visible observations.
- Mask contiguous spatial inputs for horizons 1, 2, 4, 8, and 16.
- Replay factual recorded actions and train/evaluate multi-step reward
  prediction.
- Compare correct, zeroed, and shifted action histories on controlled branches.
- Decide from evidence whether bounded Transformer context is sufficient or a
  generated state must be carried recursively.

**Gate:** masked prediction degrades gradually with horizon, reacts correctly
to branched action sequences, and beats frame/position/action-history baselines
on held-out competent trajectories.

### Checkpoint E — Actor-Critic integration

Continue with the frozen-world-model Actor/Critic checkpoint already defined in
`implementation_plan.md`. Then enable factual Critic and policy gradients
upstream block by block. Keep the model frozen during imagined Actor-only
updates and compare imagined improvement with real return.

## Performance Audit

The model is small (approximately 94k parameters), which is favourable for
memory, optimizer state, and the thesis simplicity goal. Parameter count alone
does not determine wall-clock cost.

At `64x64`, `T=16`, analytical compute is approximately:

| Component | MACs |
|---|---:|
| CNN, per frame | 21.50M |
| Token projection, per frame | 1.84M |
| Scorer, per frame | 0.061M |
| K=8 value projection/pool, per frame | 0.004M |
| Complete perception, per 16-frame sequence | 374.5M |
| Causal Transformer, per sequence | 0.905M |
| Controller reward calls, per sequence | 0.104M |

Thus roughly 99.7 percent of arithmetic is perception. This is expected: the
whole image must be inspected before informed pruning.

Top-K is currently an **information bottleneck**, not a material speedup. All
225 patches are encoded, tokenized, and scored; the selected eight are then
pooled into one 32-D vector. The Transformer always receives one vector per
time step regardless of K. All-token pooling would add only about 0.5 percent
MACs. Efficiency claims for Top-K must therefore be based on a matched measured
downstream operation, not on `8/225` alone.

Measured quick wins that preserve the research structure are:

- vectorize perception and reward calls over `(B*T)` (about 2.3--2.6x faster in
  a local GPU microbenchmark; account for changed BatchNorm statistics);
- cache the approximately 126 MB of unique raw training frames after one
  deterministic resize instead of reopening/decompressing and transforming
  each frame in overlapping windows;
- use random-offset/non-overlapping chunks so one epoch does not process each
  unique training frame about 15 times;
- use persistent data workers and benchmark batch size 32;
- log synchronized gradient norms periodically rather than every batch;
- do not enable AMP by default on the current GTX 1650, where it measured
  slower.

A standard causal mask prevents future information leakage but does not reduce
training FLOPs. The current incremental path also recomputes the full temporal
window; inference acceleration would require cached keys/values or a recurrent
state. With `T <= 20`, this is not currently the dominant cost and should remain
deferred until profiling justifies it.

## Primary Theoretical References

- Freeman, Metz, and Ha, *Learning to Predict Without Looking Ahead: World
  Models Without Forward Prediction* (2019): https://arxiv.org/abs/1910.13038
- Kingma and Welling, *Auto-Encoding Variational Bayes* (2013):
  https://arxiv.org/abs/1312.6114
- Alemi et al., *Deep Variational Information Bottleneck* (2016):
  https://arxiv.org/abs/1612.00410
- Gelada et al., *DeepMDP* (2019):
  https://proceedings.mlr.press/v97/gelada19a.html
- Grimm et al., *The Value Equivalence Principle for Model-Based Reinforcement
  Learning* (2020): https://arxiv.org/abs/2011.03506
- Locatello et al., *Challenging Common Assumptions in the Unsupervised Learning
  of Disentangled Representations* (2019):
  https://proceedings.mlr.press/v97/locatello19a.html
