# Reduced World Models — Actor-Critic Integration Design

## Purpose

This document consolidates the current architectural and training decisions for integrating an actor-critic controller into the Reduced World Models project.

The implementation gaps, evidence limits, performance audit, and checkpointed
validation protocol are maintained in `plans/architecture_validation_plan.md`.

The research goal is:

> Learn a compact, reward-oriented internal world model whose representations are sufficient for effective control, without requiring exact visual reconstruction or explicit next-state supervision.

The system remains conceptually aligned with the original World Models macro-architecture:

1. Perception / reduced representation
2. Action-conditioned temporal world model
3. Controller

The controller is extended into a compact actor-critic model with shared interpretation layers and specialized output heads.

---

## Core Research Position

The world model does not need to reconstruct the environment faithfully. It only needs to retain enough structure to:

- predict reward-relevant dynamics;
- support useful imagined trajectories;
- allow the controller to learn policies that maximize reward;
- generalize behavior to unseen situations from the same task distribution.

A representation can be imprecise and still be functionally superior.

For example, an internal assessment may behave like:

```text
environment reward:       0 + 0 + 0 + 4
dense internal progress:  1 + 1 + 1 + 1
```

The second line should not initially be interpreted as the immediate reward
prediction. It is better represented by the Critic's value estimates,
advantages, or a future explicit progress potential. The immediate reward head
must remain grounded in the environment reward because imagined trajectories
need a stable definition of return.

Silently moving reward earlier changes discounted return: receiving one unit
now is worth more than receiving the same unit later. Unconstrained reward
redistribution can therefore change the desired policy and give the Actor an
error in the learned reward model to exploit.

The robust initial design is:

- immediate reward head: predicts the observed environment reward;
- Critic: learns a dense estimate of future discounted return;
- advantage: scores whether an action was better or worse than expected;
- optional progress head or shaped reward: deferred until the factual reward
  and Actor-Critic checkpoints are stable.

Strict supervision does not require plain MSE. Sparse or heavy-tailed rewards
may use a robust distributional objective such as symlog-transformed two-hot
classification, following DreamerV3. A sequence return-consistency auxiliary
loss may later tolerate imperfect reward timing while still constraining the
sum of predicted rewards. It must remain auxiliary to an immediate reward loss
and be validated for policy invariance.

---

## High-Level Architecture

```text
RGB observation
    ↓
Reduced perception block
    ↓
Pre-action temporal belief
    ↓
Shared controller trunk
    ├── Actor head   → action distribution from b_t
    ├── Critic head  → dense internal state value V(b_t)
    └── Reward head  → predicted r_{t+1} from (b_t, a_t)
```

The three heads belong to one integrated model.

- Reward head: immediate reward prediction.
- Actor head: policy over actions.
- Critic head: expected discounted future return.

The reward head intentionally lives beside the Actor and Critic instead of
inside the temporal block. All three heads consume a common interpretation of
the pre-action belief. The reward head additionally receives the selected
current action. This gives the shared controller trunk a direct supervised
anchor while Actor and Critic objectives are changing without creating a
circular Actor input.

Architectural ownership does not change temporal semantics. The approved
transition contract is:

```text
current reduced observation p_t + previous action a_{t-1} + state z_{t-1}
    → recurrent pre-action belief z_t
    → shared controller interpretation u_t
        ├── action distribution π(a_t | z_t)
        ├── value estimate V(z_t)
        └── reward prediction R(z_t, a_t) = r_{t+1}

next step:
    p_{t+1} when visible + a_t + z_t → z_{t+1}
```

The rollout dataset stores `obs[t]` before `env.step(action[t])` and stores the
returned reward at `reward[t] = r_{t+1}`. This indexing is regression-tested.

---

## Perception Block

The complete image must be inspected before informed pruning can occur.

The goal is not to eliminate all perception cost, but to keep initial visual processing lightweight and aggressively reduce what reaches later temporal and control stages.

The perception block may contain:

- a shallow stochastic CNN encoder;
- overlapping spatial patches or tokens;
- positional encoding;
- attention scoring;
- differentiable Top-K selection;
- compact spatial pooling.

These are internal mechanisms of one block, not separate architectural models.

The selected-token ratio does not equal effective image coverage because each token may already include information from convolutional receptive fields, overlapping patches, and previous spatial aggregation.

---

## Temporal World Model

The temporal world model receives:

- current reduced perceptual representation when available;
- the previous action;
- the previous recurrent state.

Conceptually:

```text
p_t + a_{t-1} + z_{t-1}
    → MinimalSRU
    → pre-action belief z_t
```

The belief is conditioned on actions already taken. The Actor then selects
`a_t`, the reward head evaluates `(z_t, a_t)`, and `a_t` enters the recurrent
transition used to produce the next belief.

### Observational Dropout

Observational dropout is applied after spatial pruning.

At selected time steps, the temporal model continues operating without fresh
perceptual tokens. This forces `z_t` to preserve and evolve task-relevant
information using:

- previous state;
- actions.

The current hypothesis is that useful action-conditioned dynamics can emerge naturally from this pressure.

MinimalSRU recursively carries the generated state while observations are
hidden. This contract was validated under contiguous H=1/2/4/8/12 masked
horizons and through z-only imagined transitions. The former finite-context
causal Transformer is retained only on the reproducible baseline branch.

### Explicit Next-State Head

No separate next-state prediction head is required initially.

The temporal transition itself already produces the next internal state. A dedicated next-state supervision objective should be introduced only if experiments show that the implicit dynamics are insufficient.

Avoiding explicit next-state matching keeps the representation free to become functionally useful rather than visually faithful.

---

## Actor-Critic Controller

The controller is one model with:

- a shared trunk that interprets the world-state representation;
- an Actor branch;
- a Critic branch.

```text
world state
    ↓
shared trunk
    ├── actor branch
    └── critic branch
```

### Shared Trunk

The Actor and Critic both need to interpret the same latent world state.

A shared trunk:

- reduces duplicated computation;
- reduces parameter count;
- learns a common representation useful for control and evaluation.

### Separate Branches

The Actor and Critic optimize different objectives.

- Actor: choose actions.
- Critic: estimate future return.

Dedicated branches reduce interference while preserving a compact architecture.

---

## Policy

The policy is the function implemented by the Actor weights.

It maps a state to a distribution over actions:

\[
\pi(a \mid s)
\]

For continuous control, the Actor may output parameters of a probability distribution, such as mean and standard deviation.

```text
state
→ Actor weights
→ action distribution
→ sampled action
```

---

## Value Function

The Critic estimates:

\[
V^\pi(s)
=
\mathbb{E}_\pi
\left[
\sum_{k=0}^{\infty}
\gamma^k r_{t+k+1}
\mid
s_t=s
\right]
\]

Interpretation:

> The expected discounted future reward starting from state `s` and continuing with the current policy.

Where:

- `s`: current state;
- `π`: current policy;
- `r_{t+k+1}`: future rewards;
- `γ`: discount factor;
- `E`: expectation over possible future trajectories.

Two different weightings exist:

- `γ^k`: reduces the influence of temporally distant rewards;
- `E`: averages over possible future trajectories according to their probabilities.

---

## Return

For one concrete trajectory:

\[
G_t
=
\sum_{k=0}^{\infty}
\gamma^k r_{t+k+1}
\]

```text
G_t      = return from one concrete trajectory
V^π(s)   = expected value across possible trajectories
```

---

## Action-Value Function

\[
Q^\pi(s,a)
=
\mathbb{E}_\pi
\left[
G_t
\mid
s_t=s,\ a_t=a
\right]
\]

Interpretation:

> The expected discounted future reward if action `a` is forced first in state `s`, and the current policy is followed afterward.

---

## Advantage

\[
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s)
\]

Interpretation:

> How much better or worse a specific action is compared with the expected behavior from the current state.

- Positive advantage: favor the action.
- Negative advantage: reduce its probability.

This replaces the earlier idea of overfitting only to high-reward memories.

Advantage-based learning is stronger because it:

- reinforces good actions;
- suppresses bad actions;
- weights updates by how much better or worse the action was.

---

## Actor Loss

\[
L_{\text{actor}}
=
-\log \pi(a_t\mid s_t)\,
\operatorname{stopgrad}(A_t)
\]

Interpretation:

- positive `A_t`: increase action probability;
- negative `A_t`: decrease action probability;
- larger magnitude: stronger update.

The logarithm:

- converts products of probabilities into sums;
- improves numerical stability;
- provides a useful gradient form for policy optimization.

The advantage is detached so the Actor cannot reduce its loss by modifying the Critic.

---

## Critic Loss

For one-step temporal-difference learning:

\[
y_t
=
r_{t+1}
+
\gamma\,
\operatorname{stopgrad}(V(s_{t+1}))
\]

\[
L_{\text{critic}}
=
\left(
V(s_t)-y_t
\right)^2
\]

Gradient rule:

- `V(s_t)` receives gradient;
- `V(s_{t+1})` is detached because it serves as the target.

The target must remain fixed during the update. Otherwise the Critic could reduce the error by moving both its prediction and its reference.

---

## n-Step Returns

\[
G_t^{(n)}
=
r_{t+1}
+
\gamma r_{t+2}
+
\dots
+
\gamma^{n-1}r_{t+n}
+
\gamma^n V(s_{t+n})
\]

\[
A_t^{(n)}
=
G_t^{(n)}-V(s_t)
\]

Interpretation:

> Observe or imagine `n` steps, then trust the Critic again.

---

## Generalized Advantage Estimation — GAE

GAE will be used.

First define the one-step TD error:

\[
\delta_t
=
r_{t+1}
+
\gamma V(s_{t+1})
-
V(s_t)
\]

Then:

\[
A_t^{\mathrm{GAE}}
=
\delta_t
+
\gamma\lambda\delta_{t+1}
+
(\gamma\lambda)^2\delta_{t+2}
+
\dots
\]

GAE combines advantage estimates across multiple horizons.

```text
short-horizon advantages
+ medium-horizon advantages
+ long-horizon advantages
```

with progressively smaller weights.

### Role of λ

- lower `λ`: more reliance on short-horizon TD estimates;
- higher `λ`: more reliance on longer trajectories.

GAE balances lower variance from short estimates against lower bias from longer estimates.

### Impact on Losses

GAE directly provides the advantage used in the Actor loss:

\[
L_{\text{actor}}
=
-\log \pi(a_t\mid s_t)\,
\operatorname{stopgrad}(A_t^{\mathrm{GAE}})
\]

GAE can also define the Critic return target:

\[
R_t^{\mathrm{GAE}}
=
A_t^{\mathrm{GAE}}+V(s_t)
\]

\[
L_{\text{critic}}
=
\left(
V(s_t)
-
\operatorname{stopgrad}(R_t^{\mathrm{GAE}})
\right)^2
\]

Therefore, GAE affects both:

- the Actor update through the advantage;
- the Critic update through the corresponding value target.

---

## Reward Head

A direct reward-prediction head should remain.

It predicts immediate reward:

\[
\hat r_{t+1}
\]

with a direct reward loss:

\[
L_{\text{reward}}
=
\mathcal{L}(\hat r_{t+1},r_{t+1})
\]

The reward head and Critic are not redundant.

### Reward Head Responsibility

> What immediate reward follows from this state and action transition?

### Critic Responsibility

> What total discounted future reward is expected from this state under the current policy?

The Critic loss uses reward, but does not replace direct reward prediction.

The reward head is also required for imagined trajectories, where the model must produce:

\[
\hat r_{t+1}
\]

The direct reward loss provides a strong local signal and reduces dependence on a still-immature Critic.

---

## Combined Training Objectives

\[
L_{\text{total}}
=
\alpha L_{\text{reward}}
+
\beta L_{\text{critic}}
+
\eta L_{\text{actor}}
\]

The exact weights remain an implementation decision.

The main risk is not the existence of the three losses, but allowing one to dominate and damage the others.

---

## Gradient Responsibilities

This project intentionally goes beyond the safest DreamerV3 gradient boundary.
DreamerV3 is the baseline for stabilization patterns, but this thesis tests
whether behavior pressure can also shape the reduced representation. The
implementation must therefore make gradient routes explicit and enable them in
stages rather than accidentally coupling all losses at once.

### Reward Loss

Should update:

- reward head;
- shared controller trunk;
- temporal world model;
- perception block when end-to-end training is enabled.

Purpose:

> Learn representations useful for reward prediction.

### Critic Loss

Should update:

- Critic branch;
- shared controller trunk;
- temporal world model;
- potentially perception.

Purpose:

> Learn representations useful for estimating future value.

### Actor Loss

On factual real-environment objectives, may update in controlled stages:

- Actor branch;
- shared controller trunk;
- temporal world model;
- potentially perception.

Purpose:

> Shape representations that make high-reward policies easier to learn.

For an Actor objective computed from imagined learned rewards, dynamics and
reward-model parameters are frozen during the Actor optimizer step. Gradients
may pass through their operations to the chosen actions, but the Actor must not
increase its objective by changing the model or reward predictor that defines
that objective. Factual reward and Critic updates still provide the intended
end-to-end shaping. Fully coupled imagined Actor gradients remain an explicit
later ablation with reality checks, not the default contract.

This full end-to-end flow is aligned with the thesis hypothesis:

> Reward pressure should be allowed to shape the internal world representation itself.

The Actor is not only consuming a fixed world model. It may influence the representation so that the world model becomes more useful for control.

This does not imply that every Actor-Critic update must change every block from
the first experiment. The required progression is:

1. verify the world model using direct reward supervision;
2. verify Actor and Critic with the world model frozen;
3. enable Actor-Critic gradients into the temporal model;
4. enable them into spatial selection and perception with smaller learning
   rates and measured gradient norms.

At each boundary, direct reward performance, real-environment return, latent
drift, and per-loss gradient alignment must be monitored. Loss weighting is the
first stabilization mechanism. Gradient surgery or separate optimizers should
only be introduced if measured conflicts require them.

The Stage-6.0 joint-gradient audit (`src/rwm/evaluation/joint_gradient_audit.py`)
measures all six losses (visible reward MSE, masked reward MSE, tokenizer KL,
Critic, Actor, entropy) against eleven parameter blocks. It records gradient L2
norms, parameter L2 norms, ratios, and pairwise cosine similarities, and
verifies that parameter hashes, `requires_grad` flags, and optimizer state are
unchanged after measurement. The audit reconstructs SRU states from scratch
(no cached `z`) and supports two modes: eval-parity (tokenizer mean,
deterministic) and gradient-audit (seeded train mode, tokenizer sample).

---

## Imagined Trajectories

```text
b_t
→ Actor samples a_t
→ reward head predicts R(b_t, a_t) = r_{t+1}
→ temporal model advances with a_t and a missing/available p_{t+1}
→ b_{t+1}
→ repeat
```

These imagined trajectories provide:

- rewards;
- values;
- TD errors;
- GAE advantages;
- Actor targets;
- Critic targets.

No complete branching tree is required. Trajectories are sampled according to the current policy.

---

## Training Order

Actor and Critic may both be active from the beginning.

The direct reward head provides a stabilizing signal while the Critic is still immature.

Training must still control:

- relative loss weights;
- gradient magnitude;
- learning rates by block;
- rollout horizon;
- update frequency.

Changing the policy changes the meaning of:

\[
V^\pi(s)
\]

The Critic is therefore always evaluating a moving policy.

---

## Progressive Learning

The agent should continue learning after obtaining an initial functional controller because improved policies expose new states that the original world model has not seen.

The progressive cycle will reuse the validated pattern from Dreamer:

```text
collect new real experience
→ train world model on old and new experience
→ generate imagined trajectories
→ update Actor and Critic
→ use improved policy to discover new states
→ repeat
```

This reuses Dreamer’s training logic without reproducing DreamerV3 as a full architecture.

---

## Memory and Replay

The original custom memory idea is deprioritized as the primary policy-learning mechanism.

Actor-critic already provides a better method for:

- reinforcing good actions;
- suppressing bad actions;
- assigning long-term value.

Memory may still be useful as a replay and retention system.

A future replay record may contain:

- immutable source episode/file and timestep;
- policy, dataset, and world-model checkpoint provenance;
- observation or encoded scenario;
- action;
- reward;
- horizon-specific future return and terminated/truncated flags;
- historical score;
- current Critic value;
- prediction error;
- priority;
- rarity or novelty indicator.

A useful analogy is emotion:

> Some experiences should persist or be replayed more often because they were unusually rewarding, harmful, surprising, or important.

The replay policy does not need to be FIFO-only.

However, priorities should not be based only on the current Critic value because that could amplify Critic bias.

A future design may combine:

- random sampling;
- high reward;
- high negative reward;
- high TD error;
- novelty;
- rarity;
- historical significance.

### Latent replay golden rules

- **Factual pointers are permanent; latent states are disposable.** The source
  episode/timestep is the ground truth from which a current state can be
  reconstructed.
- **MinimalSRU needs only `z_t` to resume.** A cached start does not need an
  observation or temporal-history buffer during a frozen-world dream phase.
- **Every cached `z_t` is versioned by the exact world-model parameter hash.**
  Any world-model update invalidates the complete latent cache.
- **Cached starts train Actor/Critic efficiently but do not provide gradients
  into the earlier world-model path.** Joint-gradient stages must reconstruct
  `z_t` from the factual pointer with the current model.
- **Recorded behavior return is evidence, not an Actor ceiling or an unbiased
  Critic target.** The current policy may improve on it, and off-policy returns
  require explicit interpretation.
- **Sampling starts uniform.** Prioritization is introduced only after a
  measured limitation and must mix ordinary coverage with positive, negative,
  surprising, rare, and terminal experience rather than retaining only high
  rewards.
- **Capacity is bounded per stratum.** FIFO/reservoir replacement lets useful
  new experience displace old entries without erasing rare factual probes.
- **Latent replay is an efficiency layer, not a prerequisite for the first S5
  correctness and behavioral checkpoint.**

The canonical record, invalidation, gradient, and sampling rules live in
`docs/contracts/latent_memory_contract.md`. Stage scheduling remains in
`docs/plans/implementation_plan.md`, under “Optional latent-anchor replay.”

---

## Structural Simplicity

The final system remains structurally compact:

```text
Perception
→ Temporal world model
→ Shared controller
   ├── Reward head
   ├── Actor head
   └── Critic head
```

It remains one integrated three-stage system with specialized heads.

---

## Current Decisions

1. Use actor-critic.
2. Use one integrated controller with a shared trunk and specialized branches.
3. Maintain a direct reward-prediction head.
4. Keep action-conditioned temporal dynamics implicit in the temporal state.
5. Do not add an explicit next-state prediction head initially.
6. Apply observational dropout after spatial pruning.
7. Use imagined trajectories generated by the world model.
8. Use GAE for Actor advantages.
9. Use the GAE-derived return as a Critic target.
10. Allow factual reward, Critic, and real-return Actor losses to shape the
    model end-to-end in measured stages.
11. Control gradient influence through weighting rather than permanently freezing the world model.
12. Reuse Dreamer’s progressive-training pattern.
13. Deprioritize the custom memory as the main controller-learning mechanism.
14. Preserve memory as a possible prioritized replay and retention mechanism.
15. Keep task performance, generalization, and resource efficiency as the primary evaluation criteria.
16. Place the reward head beside Actor and Critic on the shared controller trunk.
17. Keep immediate environment reward and dense internal value as distinct signals.
18. Use DreamerV3 stabilization patterns as defaults, but treat end-to-end Actor-Critic gradients into the world model as an explicit thesis experiment.
19. Freeze reward/dynamics parameters during Actor-only updates based on
    imagined learned rewards; test fully coupled imagination only as a guarded
    ablation.
20. Treat an explicit next-latent/DeepMDP loss as a fallback experiment, not a
    prerequisite for observational-dropout dynamics.

---

## Remaining Implementation Decisions

These remain intentionally open:

- exact Actor distribution;
- exact shared-trunk size;
- exact private branch size;
- discount factor `γ`;
- GAE parameter `λ`;
- imagined rollout horizon;
- total loss weights;
- learning rates per architectural block;
- gradient clipping;
- exact replay-mixture weights after the uniform baseline;
- progressive update cadence;
- priority refresh cadence after a prioritized-replay limitation is measured;
- external baseline selection;
- criteria for expanding beyond `CarRacing-v3`.

These should be decided experimentally during implementation rather than fixed prematurely.
