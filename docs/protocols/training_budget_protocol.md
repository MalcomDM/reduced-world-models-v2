# Training Budget Protocol

Every result reports two non-interchangeable ledgers.

## Environment ledger

- unique collected environment transitions;
- online transitions collected for training;
- online transitions used only for evaluation;
- environment seed manifest and split.

Repeated sliding windows are not additional environment experience.

## Optimisation ledger

- world-model epochs, optimiser updates, batch size, sequence length, and
  sampled windows;
- Actor-Critic updates, warmup length, horizon distribution, factual warmup
  positions reused, and imagined transitions;
- elapsed wall time, peak GPU memory, parameter count, and throughput when
  available.

Report both ledgers before comparing methods such as DreamerV3. A gradient
update or imagined transition is not an environment interaction.

## Canonical units

- **Environment transition:** one real ``env.step(a_t)``. These are the only
  genuinely new CarRacing experience.
- **Optimiser update:** one backward pass and optimiser step. It is not a time
  step; it contains a whole batch of sequences.
- **Window:** one contiguous training sample of ``T=16`` real transitions.
  The selected world-model split has 4,040 windows per epoch.
- **Temporal position:** one frame/action position inside a window. A
  world-model epoch therefore processes ``4,040 × 16 = 64,640`` positions,
  but they overlap and are not independent transitions.
- **Controller warmup epoch:** one pass over the same 4,040 warmup windows.
  With batch size 8 this is 505 Actor-Critic updates. At warmup length 4 it
  consumes 16,160 factual perception positions. With horizons sampled
  uniformly from ``{1,2,4}``, it additionally produces about 9,427 imagined
  transitions.

The present 2,000-update Stage-5.3 run equals 3.96 controller warmup epochs,
not 2,000 real environment steps.

## Comparable reporting table

For every reference or RWM run, report:

1. real environment transitions collected for training;
2. world-model optimiser updates and replay positions processed;
3. Actor-Critic optimiser updates, factual warmup positions, and imagined
   transitions;
4. parameter counts, elapsed time, peak memory, and train/evaluation split.

DreamerV3 should be reported in these same units rather than translated to a
single fictitious "epoch": its public default specifies batch size 16,
sequence length 64, imagination length 15, and a train ratio of 32. Its exact
number of behaviour starts per batch depends on implementation details, so do
not equate its train ratio directly to RWM optimiser updates.

## Theoretical training modes

### Pure imagination controller

Actor and Critic losses use only imagined rewards and latent transitions. Real
environment interaction is restricted to collecting world-model data or
evaluation. This is the Stage 5 / Dreamer-style control claim.

### Observational-dropout joint training

Policy and world model are jointly trained in an augmented environment with
stochastic factual observations and imagined observations between them. Task
reward remains factual; observations periodically resynchronise the model.
This is the *Learning to Predict Without Looking Ahead* claim and the
controlled Stage 6 direction.

### Factual diagnostics

Real rewards may calibrate or evaluate a Critic, but the run must be named
factual and must not be presented as a policy learned purely in imagination.

## Golden rules

1. Never mix modes without recording the mode in the run configuration.
2. Keep `val` and `locked_test` seeds out of training and selection; use only
   the locked `dev` split while iterating.
3. Change one causal variable per ablation and retain resolved configuration,
   checkpoint hashes, and seed-manifest hash.
4. A higher imagined return is not real-policy improvement. Verify it in the
   real environment before making a behavioural claim.
5. Stage 5 freezes the world model. Stage 6 may unfreeze it only under the
   joint factual reward, masked-reward, KL, critic, and actor anchor schedule.
6. A persisted latent cache belongs to one exact world-model parameter hash.
   A world-model update invalidates it; the immutable rollout pointer remains
   the source of truth.
7. Count cached-state Actor/Critic work and reconstructed joint-gradient work
   separately: only the latter can update perception and the factual part of
   the temporal model.
8. Recorded behavior return is a reproducible reference, not the maximum
   achievable return and not automatically a valid target for `V^pi` after the
   policy changes.

## Current selected-run ledger

| Component | Environment ledger | Optimisation ledger |
|---|---|---|
| Masked world-model anchor, seed 42 | 5,339 collected transitions across 20 rollouts; 16 train files / 4 validation files | 93,524 parameters; 10 epochs, 5,050 batch updates (B=8), 40,400 overlapping windows of length 16; 158.7 s; peak 0.557 GB |
| Frozen Actor-Critic, Stage 5.3 | zero online training transitions; 3,000 dev evaluation transitions | 10,823 trainable parameters (Actor 5,574 + Critic 5,249); 2,000 updates, B=8, warmup 4; horizons 1/2/4 used 654/692/654 times; 64,000 reused factual warmup positions and 37,232 imagined transitions |

The world-model window positions overlap heavily; they must not be described
as 646,400 independent environment samples.
