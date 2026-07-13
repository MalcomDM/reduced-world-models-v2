# Pending Research and Engineering Work

This document records the current stopping point. It is intentionally brief: the next phase may redesign the architecture rather than preserve the current interfaces.

## Detected Technical Problems

1. **Reward gradients stop before perception.** `ReducedWorldModel._append_token()` executes under `torch.no_grad()`, detaching temporal inputs. Reward loss therefore does not train the encoder, attention scorer, selector, or spatial representation.

2. **Perception is evaluated twice per training step.** The trainer runs the encoder/tokenizer separately for KL loss and again through the complete model. This doubles work and samples two different stochastic token sets.

3. **Actions and rewards appear temporally misaligned.** Rollouts associate `obs[t]`, `action[t]`, and the resulting `reward[t]`, while training commonly predicts that reward using the previous action.

4. **Evaluation remains stochastic.** The variational tokenizer samples noise in evaluation mode, so identical inputs can produce different world states and reward estimates.

5. **The temporal-model migration is incomplete.** Manual evaluation, behavior-memory replay, and parts of controller training still call the removed LSTM `(h, c)` interface instead of Transformer history.

6. **Rollout warmup mixes NumPy and Torch actions.** The simulator can pass a NumPy action into a model path that expects tensors, preventing imagined rollout generation.

The behavior memory also has a structural limitation beyond these implementation bugs: exact hashes of rounded high-dimensional latent states are too strict to retrieve semantically similar situations.

## Abstract Continuation Steps

- Define what information a reduced world state must retain and how similarity between situations should be measured.
- Redesign memory around neighborhoods, learned similarity, clustering, or retrieval rather than exact latent identity.
- Choose an end-to-end credit-assignment strategy so reward can shape the controller, temporal dynamics, attention bottleneck, and encoder.
- Establish a minimal deterministic experiment that exposes representation drift and catastrophic forgetting before scaling training.
- Treat the current staged world-model/controller loops as experimental references, not fixed architecture requirements.

## Research Notes

The defining idea is an aggressive information bottleneck: Top-K spatial selection and observational dropout should discourage reconstruction of irrelevant detail and favor task-relevant state. This may improve representation stability, but bottlenecks alone may not solve catastrophic forgetting or credit assignment across separately trained components.

Actor-critic and Dreamer-style adversarial/value-learning approaches were previously avoided because of their complexity. They remain possible references, not assumed solutions. The desired outcome is simpler if possible: a model trainable from reward back to its earliest perception layers without requiring fragile alternating layer-wise training.
