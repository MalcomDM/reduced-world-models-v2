# Reduced World Models — Design Direction

> Execution status, current evidence, and stage gates are maintained in
> `README.md`, `plans/architecture_validation_plan.md`, and
> `plans/implementation_plan.md`. This document is the concise design rationale.

## Context

This project investigates whether a useful world model can emerge from reward pressure under strong perceptual and temporal constraints.

The main hypothesis is not that the model must reconstruct the environment accurately. Instead, it should retain only the internal structure required for the agent to perform the task satisfactorily and generalize its behavior.

The current implementation remains conceptually aligned with the original three-part World Models structure:

1. **Perception / representation**
2. **Temporal world model**
3. **Controller**

The architecture contains sophisticated internal mechanisms, but these are treated as adaptations inside the three main blocks rather than as independent models.

---

## Architectural Interpretation

### Perception block

The perception stage processes the full image because any informed spatial pruning mechanism must first inspect the complete observation.

The goal is not to eliminate all visual processing cost. The goal is to keep the initial perception stage lightweight and aggressively reduce the information passed into later, more expensive stages.

The current perception block combines:

- a shallow stochastic CNN encoder;
- spatial tokenization;
- positional information;
- attention-based scoring;
- differentiable Top-K selection;
- compact spatial pooling.

These mechanisms belong to a single perceptual block.

The selected token ratio is small, but effective image coverage is larger than the raw token percentage because of:

- convolutional receptive fields;
- overlapping patches;
- spatial aggregation before pruning.

The relevant measurement is therefore not only selected-token percentage, but also the amount of information and computational load transmitted to the temporal model.

### Temporal world-model block

Observational dropout is applied after spatial pruning.

It acts as a temporal perceptual constraint: the temporal model must continue predicting useful state and reward information even when selected perceptual tokens are unavailable at some time steps.

This is inspired by limited and discontinuous perceptual processing rather than by a literal fixed human visual frame rate.

The temporal model is a causal Transformer because a bounded causal context can
be processed in parallel during training. This advantage is only realised when
per-frame perception is batched; the current timestep loop and full-window
incremental recomputation remain explicit performance work items.

### Controller block

The controller should be adapted into a compact actor-critic model with:

- a shared internal representation;
- an actor head for action selection;
- a critic head for state-value estimation.

Actor and critic do not need to be two complete independent networks. A shared controller with two output heads is a standard and proven design.

---

## Clarified Research Position

The thesis does not require the world model to demonstrate predefined properties such as:

- accurate visual reconstruction;
- complete environment prediction;
- perfect future reward prediction;
- human-interpretable latent factors.

The primary criterion is functional sufficiency:

> Is the learned internal world model good enough for the agent to perform the task effectively and generalize to unseen situations within the same task distribution?

A model may remain imprecise while still supporting effective behavior.

The possibility of reward-specific representations or apparent reward overfitting is not merely an implementation risk. It is part of the research question inherited from the observational-dropout line of work: useful internal dynamics may emerge only to the degree required for reward maximization.

Task performance and generalization therefore remain the main validation criteria.

---

## Why Actor-Critic

Actor-critic was initially considered too complex because of its association with Dreamer.

That concern was misplaced.

The major complexity of Dreamer does not come from actor-critic itself. It comes from coordinating continual progressive learning across:

- new real experiences;
- world-model updates;
- changing latent representations;
- imagined rollouts;
- controller updates;
- replay;
- stability and catastrophic-forgetting control.

A minimal actor-critic controller is conceptually close to the existing controller and mainly requires adapting the outputs and training objectives.

Actor-critic is preferred because it provides:

- policy learning;
- value estimation;
- temporal credit assignment;
- direct optimization on imagined trajectories;
- a clean path for reward-driven end-to-end learning.

---

## Progressive Learning Strategy

The agent must continue learning after the first functional controller is obtained because improved policies will expose new states and situations that the initial world model has not learned.

This creates the central systems problem:

```text
new experiences
→ update world model
→ latent representation changes
→ controller knowledge may become invalid
→ retraining may cause forgetting
→ repeat
```

This continual-learning cycle is the main source of algorithmic complexity.

The selected direction is to reuse the validated progressive-training principles from Dreamer rather than inventing a complete new training cycle from scratch.

This does **not** mean reproducing DreamerV3 or adopting its full architecture.

The project should reuse only the abstract training pattern:

```text
collect new experience
→ train the world model using old and new experience
→ generate imagined trajectories
→ update actor and critic
→ use the improved policy to discover new states
→ repeat
```

Dreamer is used as a source of validated training conclusions, not as the architecture being implemented.

The research contribution remains the reduced world model and its reward-driven perceptual and temporal constraints.

---

## Decision on Memory

The custom temporal or behavior-memory mechanism is no longer the primary path.

Its original purpose was to:

- preserve useful projected scenarios;
- retain high-reward behavior;
- avoid catastrophic forgetting;
- reduce the need for repeated alternating training.

However, actor-critic already provides a stronger mechanism for learning which actions and states have long-term value.

A sophisticated memory introduces additional unresolved problems:

- similarity search in a changing latent space;
- persistence across encoder updates;
- semantic equivalence versus exact latent equality;
- replay bias toward only high-reward trajectories;
- additional architectural and experimental scope.

Therefore:

- memory is **partially deprioritized**;
- exact latent-state matching should not remain central;
- memory may later be reused as replay support, experience selection, or stability infrastructure;
- it should not replace actor-critic or become a second main research contribution unless later evidence shows it is necessary.

---

## Experimental Development Principle

The architecture should continue to be developed through checkpoints rather than by testing every possible component combination.

Existing checkpoint logic includes:

- validating that the encoder retains sufficient information;
- confirming that the reduced representation can support reconstruction diagnostically;
- confirming that reward prediction works through the complete world-model path;
- identifying controller integration and representation drift as the current failure point.

The project should continue to:

- start from established research components;
- adapt them intentionally;
- validate each major stage;
- compare parameter count, theoretical operations, runtime, memory use, and task performance;
- avoid exhaustive combinatorial ablations.

A small number of targeted internal comparisons may still be needed to validate the central hypothesis, but the project should not become a separate thesis for every architectural mechanism.

---

## Baseline Direction

The final external comparison should use a standard implementation such as:

- the original World Models approach; or
- DreamerV3;

depending on the final maturity and performance of the proposed agent.

The most meaningful comparison will be the trade-off between:

- task performance;
- required real-environment rollouts;
- training time;
- inference cost;
- parameter count;
- FLOPs or MACs;
- GPU memory;
- throughput;
- operational latency.

The goal is not necessarily to outperform a larger state-of-the-art model in raw reward.

The goal is to demonstrate a favorable performance-to-resource trade-off.

DreamerV3 remains an important state-of-the-art reference in world-model reinforcement learning, but should be described as a major reference or baseline rather than as the universal state of the art for all RL tasks.

---

## Current High-Level Direction

The agreed direction is:

1. Preserve the three-block World Models macrostructure.
2. Treat attention, stochastic encoding, positional information, Top-K pruning, and pooling as internal mechanisms of the perception block.
3. Keep temporal observational dropout as a core research mechanism.
4. Use a causal Transformer when its real parallel training efficiency justifies it.
5. Adapt the controller into a shared actor-critic architecture.
6. Reuse Dreamer’s validated progressive-learning cycle at an abstract level.
7. Avoid implementing Dreamer’s full architecture unless a specific mechanism becomes necessary.
8. Deprioritize the custom memory mechanism as the primary learning solution.
9. Reintroduce memory later only as replay, stability, or experience-selection support if needed.
10. Keep task performance and generalization as the principal validation criteria.
11. Expand beyond `CarRacing-v3` only after the complete training cycle is stable.

---

## Open Design Questions

These remain intentionally unresolved:

- the exact actor and critic architecture;
- the specific return estimator;
- how gradients will be propagated through the world model;
- the training cadence between real experience, world-model updates, and behavior learning;
- replay strategy and retention policy;
- which Dreamer stabilization mechanisms are actually necessary;
- whether any memory component remains useful after actor-critic integration;
- the final external baseline;
- the next environment after `CarRacing-v3`.

These decisions should be made only when the corresponding implementation problem becomes concrete.
