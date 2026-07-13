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

## 3. Implementation Mismatch (training-loop bug)

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

### What the trainer SHOULD do (deferred to Stage 2)

```
At training step t, the model should receive:
  img_t     = obs[:, t]        = s_t
  a_prev    = act[:, t]        = a_t      (current action, not previous)

Model predicts  r_pred_t
Training target r_true_t = rew[:, t] = r_{t+1}
```

This is **not** implemented here.  See §10 for the pending decision.

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

## 6. Imagination Path (`RolloutSimulator`)

The imagination path in `rollout_simulator.py`:
1. Warms up on `warmup_steps` real frames from a rollout.
2. Continues in open-loop mode: actions are sampled from the controller,
   spatial tokens are set to zeros (observational dropout in "total absence"
   mode).
3. Predicted rewards are converted to Python floats and detached.

### Input alignment

In `warmup_state()`:
```
for i in range(warmup_steps):
    img_t = obs_seq[i]            # observation at step i
    a_prev = act_seq[i]           # action taken from obs_seq[i] (numpy array)
    ...
    a_prev = a_t                  # torch version of act_seq[i] (stored but overwrites position)
```

The warmup loop correctly passes `act_seq[i]` as `a_prev`, and then stores the
same action again as `a_t`. The next iteration reads the next observation
`obs_seq[i+1]` with `a_prev = act_seq[i]` — which is the action that produced
`obs_seq[i+1]`. This means the warmup loop is internally consistent: it
processes each transition `(s_i, a_i) → s_{i+1}` by storing the action that
*led to* the next observation.

However, the first warmup step (`i=0`) uses `a_prev = act_seq[0]`. At the
model level, this feeds `a_0` as `a_prev` alongside `obs[0]` = `s_0`. But the
model's `forward` definition expects the action that was taken **before**
producing the current spatial representation — not the action taken from the
current observation. This mismatch matches the training loop shift described
in §2.

---

## 7. Evaluation Command (`evaluate_rwm_on_rollouts.py`)

The evaluation command calls:
```
h, c, r_pred, *_ = model(obs[:, t], act[:, t], h, c)
```
where `h` and `c` are `(B, WORLD_STATE_DIM)` tensors. The current model
(`ReducedWorldModel`) has signature:
```
forward(img, a_prev, history=None, lengths=None, force_keep_input=False)
```
so the positional call `model(obs[:, t], act[:, t], h, c)` maps to
`img=obs, a_prev=act, history=h, lengths=c`. Since `h` has shape
`(B, WORLD_STATE_DIM)` not `(B, T, input_dim)`, this call **fails at runtime**
with a shape mismatch inside `CausalTransformer`.

This command was written for the legacy LSTM model and has not been updated.
It is **not safe to use** with the current causal transformer.

---

## 8. Manual Test Command (`rwm_manual_test.py`)

Passes `h_prev` and `c_prev` as keyword arguments:
```
model(img=frame_tensor, a_prev=a_prev, h_prev=h_t, c_prev=c_t, ...)
```
The current `forward()` does not accept `h_prev` or `c_prev`. This raises a
`TypeError` at runtime. Same legacy LSTM incompatibility as §6.

---

## 9. Behavior Memory (`behavior_memory.py`)

Calls:
```
model.forward(img_t, a_t, h, c, force_keep_input=True)
```
where `h` = `(1, WORLD_STATE_DIM)` and `c` = `(1, WORLD_STATE_DIM)`. Same
shape mismatch as §6. These paths are broken for the causal transformer.

---

## 10. Summary of Semantic Issues

| Issue | Location | Impact |
|-------|----------|--------|
| Action shifted by 1 relative to reward target | `world_model_trainer.py:_compute_batch_loss` | Reward model may learn action-independent average |
| `done` mixes terminated/truncated | `collector.py:45` | Future bootstrap value cannot distinguish them |
| LSTM interface on Transformer model | `evaluate_rwm_on_rollouts.py`, `rwm_manual_test.py`, `behavior_memory.py` | Runtime errors |
| No validation split | `rollout_dataset.py`, `train_world_model.py` | Cannot detect overfitting |
| No deterministic probe set | — | Cannot measure latent drift |
| No transition-alignment test | — | Shift can go undetected |

---

## 11. Approved Implementation Decisions

The following decisions are approved for their specified implementation stages.

### 11.1 Canonical immediate-reward convention

**Approved decision:** Change the training loop so the model receives `a_t`
(current action) at step `t` and predicts `r_{t+1} = rew[:, t]`.

```
Desired contract:
  perceptual state s_t + selected action a_t
    → temporal transition state s_{t+1}
    → immediate reward prediction r_{t+1}

Training loop indexing (proposed):
  img_t     = obs[:, t]        = s_t
  a_prev    = act[:, t]        = a_t      (current action)
  target    = rew[:, t]        = r_{t+1}
```

This aligns the model input with the physical transition that produced the
target reward.  The current mismatch (`a_{t-1}` instead of `a_t`) is
confirmed by the deterministic spy test in
`test_transition_alignment.py::test_trainer_passes_a_prev_shifted_by_one`.

**Not implemented here.** Stage 2 will implement this approved decision.

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

**Approved decision:** Keep the current collector schema unchanged. Do not
split `done` into separate `terminated` / `truncated` flags at this stage.
The simplified `done` field is sufficient until terminal-aware training is
active, at which point the collector should be extended (not modified in
place) to store both flags.

**Not implemented here.**

---

## 12. Implementation Boundary

Stage 2 will implement the approved current-action convention for direct
reward prediction. The terminated/truncated bootstrap rule applies in Stage 4,
when Critic targets require terminal-aware value estimation. The collector
schema will be extended then rather than reinterpreting current rollout files.
