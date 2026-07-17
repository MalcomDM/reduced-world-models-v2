# Transition Contract — Stage 0 Freeze

## Purpose

This document defines the exact semantics of every transition stored in rollout
files and consumed by the training, evaluation, and imagination pipelines.
Ambiguity in these semantics makes reward-prediction results untrustworthy.
All subsequent architectural stages depend on this contract.

---

## 1. Rollout-File Layout

Each `.npz` rollout file produced by `src/rwm/data/collector.py` contains four
arrays of the same leading dimension `T` (the number of environment steps):

| Key        | Shape         | Dtype        | Semantics                              |
|------------|---------------|--------------|----------------------------------------|
| `obs`      | `(T, H, W, C)` | `uint8`     | Observation **before** executing `action[t]` |
| `action`   | `(T, A)`      | `float32`   | Action selected by the policy from `obs[t]`  |
| `reward`   | `(T,)`        | `float32`   | Reward returned by `env.step(action[t])`     |
| `done`     | `(T,)`        | `bool`       | `done or truncated` after `env.step(action[t])` |

### Collection Pseudocode (current implementation)

```
obs, _ = env.reset()
for t = 0 .. max_steps-1:
    action[t] = policy(obs)
    next_obs, reward[t], terminated, truncated, _ = env.step(action[t])
    obs[t]   = obs          (state before action[t])
    done[t]  = terminated or truncated
    obs      = next_obs
```

### Index Mapping (standard Gymnasium convention)

| Index | Meaning                          |
|-------|----------------------------------|
| `t`   | Step counter                     |
| `obs[t]` | State `s_t`                  |
| `action[t]` | Action `a_t` taken from `s_t` |
| `reward[t]` | Reward `r_{t+1}` received from taking `a_t` in `s_t` |
| `done[t]`   | Whether `s_{t+1}` is terminal |

---

## 2. Rollout-File Data Semantics (ground truth)

The data in each `.npz` file follows the standard Gymnasium
`env.step()` convention:

```
For each step t:
  action[t] = policy(obs[t])
  obs[t+1], reward[t], terminated, truncated = env.step(action[t])
  done[t]   = terminated or truncated
```

At a single index `t` the stored values represent **one transition**:

```
obs[t], action[t] → reward[t], obs[t+1], done[t]
```

This is the ground-truth convention. A forward-dynamics model should learn:

```
Given:  perceptual state s_t + action a_t
Predict:  r_{t+1} (immediate reward)
          s_{t+1} (next state, optional)
          done_{t+1} (terminal flag, optional)
```

---

## 3. Historical Implementation Mismatch (resolved in Stage 2)

This section preserves the Stage 0 bug evidence. It describes the trainer
before the Stage 2 timing correction, not the active contract.

### What the trainer actually does (`world_model_trainer.py:97`)

```
At training step t, the model receives:
  img_t   = obs[:, t]        = s_t
  a_prev  = act[:, t-1]      = a_{t-1}   (zeros when t=0)

Model predicts  r_pred_t
Training target r_true_t = rew[:, t] = r_{t+1}
```

The trainer conditions on **a_{t-1}** (the action that produced the current
state) while targeting **r_{t+1}** (the reward that resulted from a_t).
This is a one-step action shift.

### Why this is wrong

```
Rollout file index:    0        1        2        3
action:               a_0      a_1      a_2      a_3
reward (from step):   r_1      r_2      r_3      r_4

Trainer step t=0:  a_prev=0     target=r_1   (correct by accident — no prior action)
Trainer step t=1:  a_prev=a_0   target=r_2   (r_2 came from a_1, not a_0)
Trainer step t=2:  a_prev=a_1   target=r_3   (r_3 came from a_2, not a_1)
```

The model cannot learn `reward[t] = f(action[t])` because it never sees
`action[t]` at step `t`.  It must instead learn an action-independent
average or rely on spurious history correlations.  This is confirmed
deterministically by the spy-based regression test
`tests/unit/test_transition_alignment.py::test_trainer_passes_a_prev_shifted_by_one`.

### Stage 2 resolution

```
belief b_t = Transformer(obs[t], action[t-1], causal history)
action a_t = Actor(b_t)                         # future Stage 4
reward prediction = RewardHead(b_t, action[t]) = r_{t+1}
```

The previous action belongs in the pre-action belief; the current action is
supplied separately to the reward head. The active regression tests prove both
inputs and the reward target. A remaining windowing bug is that a sliding
window beginning at an interior episode offset still substitutes zero for the
true `action[offset-1]`; Stage 2.5 must return that action or use burn-in.

---

## 4. RolloutDataset Window Semantics

`RolloutDataset` (`src/rwm/data/rollout_dataset.py`) creates sliding windows:

```
window_start = offset
window_end   = offset + sequence_len
window_obs    = obs[offset : offset+sequence_len]     # s_offset .. s_{offset+seq_len-1}
window_action = action[offset : offset+sequence_len]  # a_offset .. a_{offset+seq_len-1}
window_reward = reward[offset : offset+sequence_len]  # r_{offset+1} .. r_{offset+seq_len}
window_done   = done[offset : offset+sequence_len]    # terminal_{offset+1} .. terminal_{offset+seq_len}
```

Windows containing any `done=True` are excluded by default (`include_done=False`).

### Implication for temporal batch training

When `sequence_len=16`, the window covers 16 state-action pairs. The trainer
iterates `t=0..15`, and at each step `t` the model sees one new frame.
Because `done` windows are excluded, the trainer always assumes the episode
continues; there is no terminal-state bootstrap in the current loss.

---

## 5. Done-Flag Semantics

`done[t]` is `True` when **either** `terminated` or `truncated` is `True`.

| Flag          | Meaning (Gymnasium)                                   |
|---------------|-------------------------------------------------------|
| `terminated`  | The episode reached a terminal state (e.g., crash).   |
| `truncated`   | The episode ended due to a time limit or other cut-off. |

The current code **does not distinguish** between these two cases. This matters
because:
- `terminated` → no bootstrap value; the episode truly ends.
- `truncated` → the true return is unknown; a bootstrap value from the next
  state is appropriate.

### Decision required (Stage 0 blocker)

Stage 0 does not change the done representation. The training loop currently
avoids done windows entirely, so no bootstrap logic is triggered. Stage 2
(or earlier if terminal-aware training is needed) must decide whether
terminated transitions bootstrap value differently from truncated transitions.

---

## 6. Imagination Path (`RolloutSimulator`) — inactive draft

Warmup now follows the split-action contract: observation `obs[i]` is paired
with `action[i-1]` for the belief and `action[i]` for the factual reward head.

The open-loop path is not yet an approved implementation. It currently:

1. selects `a_t` from `b_t`;
2. appends a token containing the last/zero spatial vector and `a_t`;
3. computes a new belief;
4. predicts reward using that **new** belief and the old `a_t`.

The approved order is to predict `R(b_t, a_t) = r_{t+1}` and then advance the
temporal state/context with `a_t` and the available/missing next observation.
The current code also duplicates the last real spatial vector at the first
imagined step and converts outputs to detached Python floats. Treat all of this
logic as a sketch until the Stage 2.5 masked-dynamics contract is implemented.

---

## 7. Evaluation Command (`evaluate_rwm_on_rollouts.py`)

The old LSTM evaluation implementation is now explicitly disabled with a clear
migration message. It is not an active Transformer evaluation path. The fixed
scientific evaluator defined for Stage 2.5 will replace it.

---

## 8. Manual Test Command (`rwm_manual_test.py`)

The manual command has been migrated to the active Transformer interface and
is useful for qualitative live reward plots. Its current CSV records only
episode, step, real reward, and predicted reward; it does not preserve actions,
frames, or environment seeds and is therefore not a reproducible quantitative
evaluation dataset.

---

## 9. Behavior Memory (`behavior_memory.py`)

The obsolete LSTM `recompute_keys()` path is explicitly disabled. Behavior
memory is not part of the active Stage 2 training path.

---

## 10. Summary of Semantic Issues

| Issue | Location | Impact |
|-------|----------|--------|
| Historical action/reward shift | Resolved by split belief/reward inputs in Stage 2 | Regression-tested |
| False zero previous action at interior window starts | `world_model_trainer.py:_compute_batch_loss` | Creates an artificial temporal reset per window |
| `done` mixes terminated/truncated | `collector.py:45` | Future bootstrap value cannot distinguish them |
| Draft imagination order | `rollout_simulator.py` | Reward is paired with the wrong imagined belief/action phase |
| Legacy LSTM paths | evaluation and behavior-memory modules | Explicitly disabled; manual command migrated |
| Missing terminated/truncated/seed provenance | collector schema | Blocks rigorous Critic bootstrap and deterministic evaluation |

---

## 11. Approved Implementation Decisions

The following decisions are approved for their specified implementation stages.

### 11.1 Canonical immediate-reward convention

**Implemented decision:** produce a pre-action belief from the current
observation, previous action, and causal history; supply the current action
separately to the immediate reward head.

```
belief b_t = Transformer(obs[t], action[t-1], causal history)
Actor(b_t) -> action[t]
Critic(b_t) -> V(b_t)
RewardHead(b_t, action[t]) -> reward[t] = r_{t+1}
```

This avoids circular dependence in the Actor while preserving current-action
conditioning for reward. The active spy regression test verifies previous and
current action inputs independently.

### 11.2 Terminated vs. truncated transitions

**Approved decision:**
- `terminated` transitions do **not** bootstrap.  The episode ends; no
  next-state value estimate is used.
- `truncated` transitions bootstrap **only when** a valid continuation
  state is available (i.e., when the next observation exists in the
  dataset).  If the truncated step is at the end of a recorded rollout,
  no bootstrap is applied.

This matches Gymnasium's native distinction and DreamerV3's convention.

**Not implemented here.** The current training loop avoids `done` windows
entirely, so no bootstrap logic is triggered.  This only becomes relevant
when terminal-aware Critic training begins (Stage 4).

### 11.3 Rollout-file schema and done representation

**Historical decision:** the Stage 0 collector was left unchanged. Stage 2.5
now requires a versioned schema extension with separate `terminated` and
`truncated`, environment seed, policy provenance, and collector configuration
before terminal-aware Critic training or final evaluation data are collected.

---

## 12. Implementation Boundary

Stage 2 implemented the split pre-action-belief/current-action reward contract.
Stage 2.5 fixes interior-window context, provenance, and masked dynamics before
the Critic or imagination paths become active. Existing rollout files retain
their historical schema and are never silently reinterpreted.
