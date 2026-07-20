# SRU Cell Equation Analysis — Stage 0 Complete

**Status:** **APPROVED — MinimalSRUTemporal selected**
**Author:** Stage 0 temporal contract audit
**Date:** 2026-07-19 (corrected 2026-07-19)

---

## Option 1 — MinimalSRUTemporal (SELECTED)

### Exact equations

```
Input:      x_t = concat(observation_keep_t * spatial_t, previous_action_t, observation_keep_t)
                                                              total: R^{36}
Projection: p_t = W_p @ x_t + b_p                      # Linear(36 → 160)
candidate_t  = tanh(p_t[..., :80])                     # (B, 80)
carry_t      = sigmoid(p_t[..., 80:] + carry_bias_init) # (B, 80); carry_bias_init = 1.0
z_candidate  = carry_t * z_{t-1} + (1 - carry_t) * candidate_t
z_t          = where(valid_step_t, z_candidate, z_{t-1})  # padding guard
```

Where:
- `z_t ∈ R^{80}` — recurrent state (also the complete temporal belief).
- `z_0 = zeros(B, 80)` at episode start.
- `carry_bias_init = 1.0` initialises the carry gate near 1, preserving state
  early in training.  Termed `carry_bias` not `forget_bias` to avoid conflating
  this simple gate with an LSTM forget gate.
- `valid_step_t` is a boolean per step; `True` = normal update, `False` = padding
  (state unchanged).  This is **separate** from `observation_keep_t`:
  `observation_keep=False` means the spatial input is zeroed but actions and the
  visibility bit remain, so `z_t` still updates.

### Parameter count and MACs

| Component | Parameters | MACs (per step) |
|-----------|-----------|-----------------|
| `W_p` | 36 × 160 = 5,760 | 5,760 |
| `b_p` | 160 | — |
| Tanh (80 units) | 0 | 80 |
| Sigmoid (80 units) | 0 | 80 |
| Elementwise gates | 0 | 2 × 80 = 160 |
| **Total** | **5,920** | **6,080** |

MACs per training step (200-step blind horizon): 6,080 × 200 = 1.216M.
Compare: causal Transformer at T=20 is ~1.14M MACs for the entire 20-token
window.  Over 200 blind steps the SRU is 200× cheaper than the Transformer
(which would need to reprocess its full window at every step).

### Recurrent state required for resume

- **Only `z_t`** — one `(B, 80)` tensor.
- No history, no lengths, no KV cache.

### Train-time parallelisable vs sequential operations

- **Sequential:** the recurrence `z_t = f(x_t, z_{t-1})` is a serial dependency
  across the time dimension.  No `T`-level parallelism.
- **Parallelisable across batch:** perfect (each batch element independent).
- Within the cell, the two halves of the projection and the two gates are
  computed in one fused `Linear(36 → 160)` call → fully parallel.

### Inference cost

- 6,080 multiply-accumulate operations per step.
- At 200 Hz (5 ms per step) on a single CPU core: negligible.
- GPU: one tiny `Linear` kernel launch → dominated by launch overhead, not
  arithmetic.

### Stability risks

- **Forget-gate saturation:** with `forget_bias = 1.0`, the gate starts
  saturated near 1.  No vanishing gradient for the recurrent state (the
  additive `(1 - forget) * candidate` path is small).  This is the standard
  LSTM/GRU trick.
- **No output gate or reset:** the entire state is exposed to the controller.
  This is an intentional simplification — directly satisfies the thesis
  requirement that `z_t` *is* the dynamic latent state.
- **No normalisation:** no layer-norm on `z_t`.  If reward or KL losses push
  `z_t` values outside the tanh range, gradients may destabilise.  Add optional
  `LayerNorm(80)` after the cell if instability is observed in S2.

### Controller input and recurrent state identity

- **They are identical.** `z_t` is both the recurrent state and the input to
  `ControllerTrunk.encode()`.  This satisfies the thesis requirement directly.
- No separate "belief" extraction step (contrast with the causal Transformer
  which extracts the last valid position from a full sequence).
- **Note on canonical SRU:** A canonical SRU (Lei et al. 2017) can also be
  configured so that the controller input equals the recurrent state (e.g., by
  omitting the output gate or reading from the pre-output state).  The earlier
  S0A claim that a canonical SRU "inherently breaks state identity" was
  overstated.  The correct statement is: this project **deliberately selects**
  the smaller MinimalSRUTemporal cell for simplicity, adequate capacity, and
  single-gate stability.  The canonical SRU was rejected on pragmatic grounds
  (more parameters, three saturable gates, unnecessary complexity for the
  ~94k-parameter total model budget), not because it fundamentally cannot
  satisfy the thesis requirement.

### Thesis compatibility

- `z_t` is explicitly the dynamic latent state carried forward — exactly what
  the thesis requires.
- The `observation_keep_t` bit in `x_t` lets the cell distinguish masking from
  reset without hard-coded logic.

---

## Option 2 — Canonical SRU-Style Cell

### Exact equations

```
Input:      x_t ∈ R^{36}
Projection: p_t = W_p @ x_t + b_p           # Linear(36 → 4*80 + skip_dim)
                                         # skip_dim discussed below

# Candidate state
candidate_t = tanh(p_t[..., :80])

# Gates (all sigmoid, same projection slice)
forget_t   = sigmoid(p_t[..., 80:160] + forget_bias)
reset_t    = sigmoid(p_t[..., 160:240])
output_t   = sigmoid(p_t[..., 240:320])

# Recurrent step (with reset gate)
h_t        = (1 - reset_t) * z_{t-1} + reset_t * candidate_t   # "reset" blends old and new
# or equivalently: h_t = candidate_t * reset_t + z_{t-1} * (1 - reset_t)

# Output gate
z_t        = output_t * tanh(h_t)

# Optional skip connection (not standard SRU but common in later variants)
# z_t = z_t + W_skip @ x_t    # adds another Linear(36 → 80)
```

Note: the "canonical SRU" (Lei et al., 2017) uses a light recurrence that is
primarily a highway-style skip.  The gates above follow a GRU-style layout.
A true canonical SRU has:

```
f_t = sigmoid(W_f @ x_t + b_f)
r_t = sigmoid(W_r @ x_t + b_r)
c_t = f_t * z_{t-1} + (1 - f_t) * (W_c @ x_t)
z_t = r_t * tanh(c_t) + (1 - r_t) * x_t    # highway output
```

Which is closer to Option 1 but with a separate output/reset gate and a
projected skip path.

### Parameter count and MACs

For the GRU-style layout (candidate + 3 gates, no skip projection):

| Component | Parameters | MACs (per step) |
|-----------|-----------|-----------------|
| `W_p` (36 → 320) | 36 × 320 = 11,520 | 11,520 |
| `b_p` | 320 | — |
| Tanh (80) + 3×Sigmoid (80) | 0 | 320 |
| Elementwise (gates, reset, output) | 0 | 4 × 80 + 2 × 80 = 480 |
| **Total** | **11,840** | **12,320** |

For the canonical SRU (Lei et al.) with highway output:

| Component | Parameters | MACs (per step) |
|-----------|-----------|-----------------|
| 3 × `Linear(36 → 80)` or fused `Linear(36 → 240)` | 36 × 240 = 8,640 | 8,640 |
| `W_c @ x_t` | 36 × 80 = 2,880 | 2,880 |
| Biases | 240 + 80 = 320 | — |
| Elementwise | — | 4 × 80 + 2 × 80 = 480 |
| **Total** | **11,840** | **11,520** (with fused gate proj) |

### Recurrent state required for resume

- `z_t` — `(B, 80)` — same as Option 1 for the Lei-style SRU.
- The GRU-style cell with `h_t` and `z_t` still only needs `z_t` (output)
  for the next step, because `h_t` is fully determined from `z_{t-1}` and `x_t`.

### Train-time parallelisable vs sequential operations

- Same as Option 1: sequential over time, parallel over batch.
- Slightly more sequential elementwise work per step (3 gates vs 1).

### Inference cost

- Approximately 1.9–2.0× Option 1 MACs per step (11,520 vs 6,080).
- Still dominated by kernel launch overhead on GPU.

### Stability risks

- More gates mean more saturable nonlinearities.  The reset and output gates
  add two additional sigmoid saturation risks.
- The output gate `output_t * tanh(h_t)` introduces a multiplicative gating
  on the belief before the controller sees it.  This means the controller
  input is no longer the raw recurrent state — violating the thesis requirement
  that `z_t` itself be the complete dynamic latent state.
- The highway path (`(1 - r_t) * x_t`) in the Lei-style SRU mixes the raw
  input `x_t` directly into the state, which means `z_t` contains the raw
  `observation_keep_t` bit and `previous_action_t` in a skip path that bypasses
  the nonlinear transform.  This makes the state partially linear and harder
  to interpret as a learned representation.

### Controller input and recurrent state identity

- **Not identical for GRU-style:** `z_t = output_t * tanh(h_t)`.  The output
  gate suppresses some dimensions.  If `output_t` is near 0 for some dimension,
  the controller receives a gated value, not the full recurrent state.
- **Not identical for Lei-style SRU:** `z_t = r_t * tanh(c_t) + (1 - r_t) * x_t`.
  The highway skip copies raw input into `z_t`.
- **Neither satisfies the "z_t is the state" requirement** as cleanly as
  Option 1.  An additional "belief extraction" step would be needed if
  interpretability matters.

### Thesis compatibility

- The output gate and highway skip break the direct identity between recurrent
  state and controller input.  The thesis states "z_t must be the complete
  state required for the next temporal step."  An output-gated cell stores
  more information in `z_t` than it exposes — the controller sees a gated
  subset, but the next step uses the full ungated state.

---

## Comparison Summary

| Property | Option 1 — Minimal SRU-Like | Option 2 — Canonical SRU |
|----------|---------------------------|--------------------------|
| Parameters | 5,920 | 11,840 |
| MACs per step | 6,080 | 11,520–12,320 |
| Recurrent state | `z_t (80)` | `z_t (80)` |
| Resume state | `z_t` only | `z_t` only |
| Controller input = state? | **Yes** — identical | **No** — output-gated or highway-skip |
| Thesis "z_t is dynamic state" | ✅ Satisfied directly | ❌ Broken by output/highway |
| Stability risk | Low (1 saturable gate) | Medium (3 saturable gates) |
| Representational capacity | Lower (1 gate) | Higher (3 gates) |
| Inference cost relative | 1.0× | ~1.9× |
| Implementation complexity | Trivial | Moderate |

---

## Decision

**APPROVED — MinimalSRUTemporal selected.**

Rationale:

1. **Thesis requirement:** `z_t` must be the complete dynamic latent state.
   MinimalSRUTemporal satisfies this directly.  (Note: a canonical SRU could
   also satisfy this with appropriate gate configuration; the earlier S0A
   claim that it "inherently breaks state identity" was overstated.  The
   correct rationale is pragmatic, not categorical.)

2. **Simplicity:** 5,920 parameters vs 11,840.  One gate vs three.  The
   minimal cell is easier to debug, profile, and reason about.

3. **Adequate capacity:** The causal Transformer has 56,560 temporal
   parameters.  Even the minimal SRU is not capacity-limited — the bottleneck
   is perception, not the temporal backbone.  Extra gates are unlikely to
   improve reward prediction given the ~94k total model parameters.

4. **Stability:** One carry gate is easier to regularise than three.
   The `carry_bias_init = 1.0` trick is well-understood and reliable.

5. **Naming:** Called `MinimalSRUTemporal` in code and documentation.
   Not claimed to be the canonical SRU (Lei et al. 2017) — the architecture
   is inspired by the SRU family but is simpler.
