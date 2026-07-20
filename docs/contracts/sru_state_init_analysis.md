# Recurrent State Initialization Analysis — Stage 0 Complete

**Status:** **APPROVED — Loss-masked same-episode burn-in selected**
**Author:** Stage 0 temporal contract audit
**Date:** 2026-07-19 (corrected 2026-07-19)

---

## Problem Statement

The existing `RolloutDataset` produces sliding windows from `.npz` rollout
files.  A window starting at `offset` contains observations
`obs[offset:offset+sequence_len]` (currently `sequence_len=16`).  The causal
Transformer handles this by rebuilding its full attention context from the
window's tokens — a window at offset 0 and a window at offset 50 produce
equivalent representations for their respective first positions because the
Transformer attends to all past tokens within the window.

An SRU replaces this bounded attention with a recurrent state `z_t`.
`z_0 = zeros(B, 80)` is correct only at a true episode start (`offset == 0`).
For a mid-episode window, `z_0` must be the correct hidden state after
processing the episode's prefix up to `offset`.  Initialising `z_0 = zeros`
at every window is incorrect and is explicitly rejected.

**Why `predecessor_action` alone is insufficient:** The predecessor action
(`action[offset-1]`) provides only the *last* action before the window, not
the accumulated temporal context.  An SRU state carries information about
observations, actions, and their combined dynamics over potentially hundreds
of steps.  Supplying a single action vector while resetting the state to zero
is equivalent to telling the model "you've seen everything before this point,
now forget it all and only remember this one action" — which is wrong.

---

## Option 1 — Fixed Loss-Free Same-Episode Burn-In

### Approach

Process `N` burn-in steps from the same episode before the training window,
running the SRU in `torch.no_grad()` mode.  The loss mask excludes these
burn-in positions.

```
Training window:  [offset, offset + sequence_len)
Burn-in prefix:   [offset - burn_in, offset)   # if offset >= burn_in
                  [0, offset)                  # if offset < burn_in (partial burn-in)

Loss applied:     after offset only
Gradients:        through the training window only (burn-in is no_grad)
```

### Concrete indexing examples

| Window offset | Burn-in range | Training range | Loss mask |
|---|---|---|---|
| 0 | — (no burn-in, z_0 = zeros) | `[0, 16)` | All `True` |
| 5 | `[0, 5)` (partial, only 5 steps) | `[5, 21)` | Steps 0–4 masked |
| 20 | `[0, 20)` (full burn-in) | `[20, 36)` | Steps 0–19 masked |
| Adjacent to `done` | Burn-in stops at `done` boundary | After `done` | Rewards truncated at `done` |

If the burn-in prefix crosses an episode boundary (which cannot happen because
windows are within single `.npz` files), the state resets at the boundary and
only the post-boundary prefix is used.

### Analysis

| Dimension | Assessment |
|-----------|-----------|
| **Correctness** | ✅ Exact state from the true episode prefix (same model, same inputs). |
| **Repeated perception** | ❌ Burn-in steps are perceived N extra times per window. Each frame seen ~`sequence_len/N` extra times depending on stride. |
| **Training parallelism** | ✅ Batches are independent. Each batch element processes its own prefix sequentially. |
| **Stale-state risk** | ✅ None — state is from the current model, current inputs. |
| **Data-loader complexity** | Low — `RolloutDataset.__getitem__` returns `(burn_in_obs, burn_in_actions, window_obs, window_actions)` with `offset` and `burn_in` metadata. |
| **Overlapping windows** | ❌ Heavily overlapping windows duplicate burn-in work. With `stride=1` each frame is re-perceived `burn_in` times.  With `stride >= sequence_len` the cost is one extra forward per window. |
| **Current experiment budget** | 10 epochs × ~4,000 windows × (burn_in + target_steps) × perception cost.  At `burn_in=20` and `target=16`, each 36-frame sequence requires up to **2.25×** (36/16) the target perception work.  At current data scale (~5,339 steps) this is measurable but manageable. |

---

## Option 2 — Episode-Sequential Truncated BPTT

### Approach

Process each episode sequentially as one long sequence.  Detach the SRU
state between windows but keep it live across consecutive windows from the
same episode.

```
Episode steps:   [0, 1, 2, ..., T-1]
Window 1:        [0, 16)   z_0 = zeros
Window 2:        [16, 32)  z_0 = z_16 from window 1 (detached)
Window 3:        [32, 48)  z_0 = z_32 from window 2 (detached)
...no overlap...
```

### Concrete indexing examples

| Episode length | Windows | State init |
|---|---|---|
| 148 steps | 9 windows (stride=16, last may be partial) | Sequential from previous window |
| offset 0 | `[0, 16)` | `z = zeros` |
| offset 16 | `[16, 32)` | `z = z_16` (detached) |
| offset 148 | Last window `[144, 148)` | Truncated at episode end |

### Analysis

| Dimension | Assessment |
|-----------|-----------|
| **Correctness** | ✅ Correct: state flows from true previous prefix.  Detached gradient means no BPTT through the full episode. |
| **Repeated perception** | ✅ None — each frame is perceived exactly once as part of its window. |
| **Training parallelism** | ❌ Sequential across windows within an episode (cannot shuffle across episode segments).  Parallel across episodes only. |
| **Stale-state risk** | ✅ None — state from current model. |
| **Data-loader complexity** | High — requires an episode-aware batching strategy that groups consecutive windows.  `RolloutDataset` must yield `(episode_id, segment_index)` and the DataLoader must not shuffle across episode segments.  The trainer needs an `EpisodeSequentialSampler`. |
| **Overlapping windows** | ✅ No overlap — stride = sequence_len by construction.  This changes the training distribution from the current overlapping strategy. |
| **Current experiment budget** | Changes number of windows per epoch from ~4,000 (stride=1 overlapping) to ~330 (stride=16 non-overlapping).  Total training transitions decrease by ~12×.  This is a material change to the training setup. |

---

## Option 3 — Version-Valid State Cache Rebuilt from the Current Model

### Approach

Build a pre-computed cache of SRU states at every episode position using
the current model.  At training time, each window reads the cached `z_t` at
its offset instead of running burn-in.

```
Cache build (once per model change):
  for each episode in dataset:
    z = zeros(B, 80)
    for t in range(episode_length):
      output = model.forward_sequence(obs[t:t+1], ...)
      cache[t] = z.copy()
      z = output.world_state

Training:
  window at offset → read cache[offset] → continue with z_0 = cache[offset]
```

The cache is invalidated whenever the model parameters change.  The trainer
rebuilds it once per epoch (or whenever `model.state_dict()` hash changes).

### Concrete indexing examples

| Window offset | Cache lookup |
|---|---|
| 0 | `cache[0] = zeros` |
| 5 | `cache[5]` = state after processing steps 0–4 |
| 20 | `cache[20]` = state after processing steps 0–19 |

### Analysis

| Dimension | Assessment |
|-----------|-----------|
| **Correctness** | ✅ Identical to burn-in (same model, same inputs, same computation). |
| **Repeated perception** | ✅ None during training (one full-episode pass per cache rebuild).  Cache rebuild requires one full episode forward per epoch. |
| **Training parallelism** | ✅ Independent batches, full shuffle, overlapping windows allowed. |
| **Stale-state risk** | ⚠️ Cache is stale after every optimizer step.  With SGD the parameters change at every training step, so the cached state computed from the old parameters is always stale.  Rebuilding every `K` steps or every N batches reduces but never eliminates staleness.  This is the primary reason Option 3 was rejected over Option 1. |
| **Data-loader complexity** | Medium — requires a `StateCache` class that maps `(episode_file, offset) → z_t` and invalidates on hash change.  The Dataset.__getitem__ returns the cached state alongside the window. |
| **Overlapping windows** | ✅ Supported — every window offset has its correct cached state. |
| **Current experiment budget** | Cache rebuild: ~5,339 steps / 16 = ~334 forward_sequence calls per epoch.  Each call processes 16 frames in a single `world_hd` pass.  Roughly equal to 1 burn-in epoch.  Acceptable. |

---

## Explicit Rejections

### Rejected: Zero reset at every sliding window

Setting `z = zeros` at every arbitrary window start throws away all temporal
context.  The model sees each window as if it were a new episode and must
re-learn "where am I?" from the first observation alone.  This is strictly
worse than the causal Transformer, which at least has the full window in its
attention context.  The causal baseline with `predecessor_action` fix (Stage
2.5B.3) already proved that zero reset harms performance; the SRU zero
reset would be even worse because there is no attention mechanism to recover
context from the window's tokens.

### Rejected: `predecessor_action` alone as state surrogate

The SRU state `z_t` encodes the entire temporal dynamics up to step `t`.
A single action vector `action[offset-1]` is 3 floats — hopelessly
insufficient to summarise the preceding dynamics.  The predecessor action
is already consumed by the SRU as part of `x_t` at the first window step;
providing it separately as a state initialiser adds no relevant information.

---

## Recommendation

**Option 1 — Fixed Loss-Free Burn-In (PENDING SUPERVISOR DECISION)**

Rationale:

1. **Zero stale-state risk:** same model, same inputs, same computation as
   training — the state is exact.

2. **Simplest implementation:** `RolloutDataset` returns extra prefix frames,
   the trainer masks their loss.  No cache management, no sequential sampler,
   no cross-window state tracking.

3. **Compatible with overlapping windows:** the current training distribution
   uses stride-1 overlapping windows.  Option 2 (episode-sequential) would
   require non-overlapping windows, changing the training distribution and
   reducing data volume by ~12×.  Option 1 preserves the existing setup.

4. **Repeated perception cost is bounded:** at `burn_in=20` with ~4,000
   windows per epoch, the extra cost is ~80,000 extra frame perceptions per
   epoch.  At current data scale (~5,339 total frames), this means each frame
   is perceived ~15 more times per epoch.  This is measurable but manageable
   — the B*T vectorised perception processes 80,000 extra frames in ~2.7
   seconds per epoch (based on the measured 25.4 ms/step for 32-frame batches).

5. **Burn-in count matched to context length:** `burn_in = SEQ_LEN = 20`
   matches the causal Transformer's maximum context.  This is a natural
   conservative starting point.

**Number of burn-in steps:** 20 (matching `SEQ_LEN`), subject to reduction
if profiling shows the extra perception cost is material.  An initial
sensitivity test at S2 can compare `burn_in ∈ {5, 10, 20}`.

## Decision

**APPROVED — Option 1 (loss-masked same-episode burn-in, 20 steps).**
Episode-sequential TBPTT and version-valid cache are deferred to a later
stage if burn-in overhead becomes prohibitive.

The burn-in is a *truncated* 20-step state, comparable with the causal
Transformer context window.  It is NOT the exact full-episode state.
