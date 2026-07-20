# MinimalSRU Temporal Contract

**Status: IMPLEMENTED AND SELECTED.** MinimalSRU was adopted as the primary
temporal backend after the completed S0–S6 validation line. The causal
Transformer remains only as a reproducible compatibility/reference backend
and on `baseline/causal-transformer-stage5`.

## Purpose

Define the exact interface, tensor shapes, timing semantics, state-carry
rules, and causal-backend compatibility requirements for replacing the
bounded-history `CausalTransformer` with a compact recurrent SRU-like cell.
This document freezes the contract **before** implementation so that the
architecture decision is explicit and the two backends can coexist without
changing existing training behaviour.

## Approved Decisions (S0 Gate)

The following two supervisor decisions are now locked and recorded:

1. **Temporal cell:** MinimalSRUTemporal (not canonical SRU). Rationale:
   simplicity, thesis requirement that `z_t` is the complete dynamic latent
   state, adequate capacity, and single-gate stability.  Named
   `MinimalSRUTemporal` or `SRULikeTemporal` in code.  Not claimed to be
   the canonical SRU (Lei et al. 2017).

2. **Mid-window state initialization:** 20-step loss-masked same-episode
   burn-in (not episode-sequential TBPTT, not version-valid cache).
   This is a *truncated* 20-step initial state, comparable with the causal
   Transformer context window.  It is NOT the exact full-episode state.
   Zero-state reset at arbitrary sliding windows and `predecessor_action`
   alone as a state surrogate are explicitly rejected.

See `docs/plans/sru_temporal_validation_plan.md` for the staged gate
structure.  Stage 0 milestones are complete.

---

## 1. Tensor Shapes and Timing

### Input token `x_t`

```
x_t = concat(
    observation_keep_t * spatial_t,    # (B, 32) — values_dim, spatial representation
    previous_action_t,                 # (B, 3)  — action[t-1]; zeros at t=0
    observation_keep_t,                # (B, 1)  — explicit visibility bit
)                                      # total (B, 36)
```

- `spatial_t`: output of `SpatialAttentionHead` after `ObservationalDropout`.
- `previous_action_t`: `action[t-1]`; zeros when `t == 0`.
- `observation_keep_t`: `1.0` for visible, `0.0` for masked (broadcast from bool).
  This bit is consumed only by the SRU cell; the causal Transformer already
  receives it implicitly via left-truncation of zeroed spatial tokens.
- The explicit `observation_keep_t` bit ensures the SRU can distinguish
  "masked observation, carry on" from "episode start, reset state".

### Recurrent state `z_t`

```
z_t = TemporalStep(x_t, z_{t-1})    # (B, 80)
```

- `z_t` is the **complete** temporal state required for the next incremental step.
- No auxiliary carry (no separate hidden/cell, no KV cache, no history buffer).
- `z_0 = zeros(B, 80)` at episode start (before any observation).

### Reward timing

```
reward_pred_t = ControllerTrunk.encode(z_t)  → shared_repr
ControllerTrunk.predict_reward(shared_repr, current_action_t)  → reward_t
```

- `current_action_t = action[t]` (the action being evaluated, not the previous one).
- Identical to the existing causal contract.

### Actor/Critic consumption

```
shared_repr = ControllerTrunk.encode(z_t)
Actor(shared_repr)    → action[t]
Critic(shared_repr)   → V(z_t)
```

- Identical to the existing causal contract.

---

## 2. Sequence and Incremental APIs

### `forward_sequence` (full-sequence training)

```python
# Vectorised perception: (B, T, C, H, W) → (B, T, 36) input tokens.
z_0 = torch.zeros(B, 80, device=device)
z_all = []  # (B, T, 80)
for t in range(T):
    z_t = temporal_cell(x_t[:, t, :], z_{t-1})
    z_all.append(z_t)
z_all = torch.stack(z_all, dim=1)

# Vectorised controller: (B*T, 80) + (B*T, 3) → (B*T, 1)
# Same as causal path.
```

Implementation note: the loop is a `for` over `T` steps (sequential scan during
training), identical to the current `forward_sequence` which runs perception
vectorised but the temporal backbone sequentially.  Future parallel-scan
optimisation is a separate concern.

**Contract:** The fixed-input-dimension SRU cell (no `max_seq_len`, no `SEQ_LEN`
truncation) replaces the Transformer's `return_all=True` path.  `all_out` is
`z_all`.  `reward_pred_seq` is computed identically.

### `forward` (incremental inference)

```python
# Single step: perceive frame → build x_t
z_t = temporal_cell(x_t, z_{t-1})  # z_{t-1} from previous call
```

**Contract:** No `history`, `lengths`, or `HistoryBuffer` arguments are needed.
**However**, for backward compatibility during S0–S1 coexistence both backends
must accept the existing signatures.  The SRU will ignore `history` and
`lengths` when provided, carrying only `z_{t-1}`.

### `WorldModelOutput` for SRU mode

```python
class WorldModelOutput(NamedTuple):
    world_state: Tensor          # (B, 80) — z_t (identical semantics)
    reward_pred: Tensor          # (B, 1)
    mask_soft: Tensor            # (B, N) — unchanged
    indices: Tensor              # (B, K) — unchanged
    history: Tensor              # (B, 1, 36) — placeholder token only (compatibility)
    lengths: Tensor              # (B,) — all-ones (compatibility)
    tok_mu: Optional[Tensor]    # unchanged
    tok_logvar: Optional[Tensor] # unchanged
    reward_pred_seq: Optional[Tensor]  # unchanged
```

- `history` and `lengths` are populated only for backward compatibility with
  callers that unpack them for the next incremental call.  The SRU forwards
  `z_t` directly; the output `history` is a single-token buffer containing
  merely the current `x_t` frame.  Lengths is `ones(B,)`.
- All callers that currently read `out.history` and `out.lengths` for the
  **next** call must be updated to read `out.world_state` instead in S1.
  During S0 coexistence the causal path continues as before.

---

## 3. Reset, Padding, Masking, and Episode-Boundary Semantics

### Reset

- **Episode start:** `z = zeros(B, 80)` before any observation.
- **No reset at sliding-window boundaries** (see initialization analysis below).
- **No reset at `done`:** the dataset excludes done windows; the trainer never
  sees terminal transitions.

### Padding (`valid_step` mask)

- `valid_step=False` means this time position is padding (no valid input).
  The cell must leave the state unchanged: `z_t = z_{t-1}` regardless of
  input values.
- Invalid padding does not occur in the current `RolloutDataset` because
  windows are contained within single `.npz` files and `done` windows are
  excluded.  The `valid_step` mask is available for future use (e.g.,
  variable-length episodes, packed sequences).
- Implemented as `z_t = where(valid_step_t, z_candidate, z_{t-1})`.

### Masking (observational dropout, `observation_keep`)

- **Critical semantic distinction:** `observation_keep` and `valid_step` are
  separate masks and must never be conflated.
- `observation_keep=False` means "eyes closed": the spatial input is zeroed,
  but `previous_action` and the visibility bit remain available.  The recurrent
  state **must update** (z changes) because the action provides temporal
  continuity.
- When `observation_keep_t = False`:
  - `spatial_t` = zeros (already guaranteed by `forward_sequence` multiplying by `keep_float`).
  - `previous_action_t` = actual `action[t-1]` (unmasked — actions remain visible).
  - `observation_keep_t` = `0.0` (explicit bit).
- The SRU receives the mask bit explicitly in the input `x_t`.  The cell can
  learn to modulate carry/candidate contributions when the bit is 0.

### Episode boundaries

- State resets only at real episode starts (`offset == 0` in the dataset).
- No synthetic zero-state reset at sliding-window boundaries.
- The initial `z_0` for a mid-episode window is obtained from a 20-step
  loss-masked same-episode burn-in.  This is a *truncated* 20-step initial
  state, comparable with the causal Transformer's 20-token context window.
  It is NOT the exact full-episode state.
- Burn-in cost: a full 20 (burn-in) + 16 (target) = 36 frame sequence requires
  up to **2.25×** the target perception work (36/16), because the burn-in
  prefix must be perceived alongside the target window.
- **Gradient policy (S2 approved):** Direct MSE/KL loss is excluded from
  burn-in positions.  Target loss gradients **flow through** the burn-in state
  construction — no ``torch.no_grad()`` or ``detach()`` at the burn-in/target
  boundary.  This is the required end-to-end reward-learning path.
- **Processed-frame accounting:**
  - Target positions: 16 (where loss is computed).
  - Total positions: 36 (where the model computes states).
  - Burn-in positions: 20 (state construction only, no direct loss).

---

## 4. Z-Only Save/Resume Semantics

### Save

Checkpoints store `model.state_dict()` which includes the SRU cell parameters.
No additional per-step cache is saved — the SRU has no context window to
serialise.

**Causal checkpoint keys that disappear:**

| Key prefix | Status |
|---|---|
| `world_hd.input_proj.weight` | Removed |
| `world_hd.input_proj.bias` | Removed |
| `world_hd.pos_emb.weight` | Removed |
| `world_hd.encoder.layers.0.*` | Removed |

**SRU checkpoint keys added:**

| Key prefix | Status |
|---|---|
| `world_hd.cell.*` | Added |

### Resume

- `z_t` is not part of `state_dict`.  On resume the first call starts from
  `z = zeros(B, 80)` (episode start) or from a cached `z` (see initialization).
- Causal checkpoints continue to load unchanged through `load_checkpoint()`
  with a warning about the unmatched `world_hd` keys.
- The `model_from_checkpoint()` factory selects the backend from
  `TemporalConfig.temporal_backend` ("causal_transformer" | "sru").

### Round-trip compatibility

- Loading a causal checkpoint into an SRU model **fails** (world_hd key mismatch).
- Loading an SRU checkpoint into a causal model **fails** (missing pos_emb etc.).
- Loading a legacy bare `state_dict` into either backend:
  - When `TemporalConfig.temporal_backend == "causal_transformer"` — unchanged
    behaviour (default, backward compatible).
  - When `TemporalConfig.temporal_backend == "sru"` — raises a clear error
    because `world_hd` keys do not match.

---

## 5. Causal-Backend Compatibility Requirements

During S0–S5 coexistence the two backends must:

1. Share the same `ReducedWorldModel` constructor signature.
2. Accept the same `forward()` and `forward_sequence()` signatures (the SRU
   ignores `history`, `lengths`).
3. Produce `WorldModelOutput` with the same named fields.
4. Use the same `ControllerTrunk`, perception stack, and loss functions.
5. Support `TemporalConfig.temporal_backend` in `model_from_checkpoint()`.
6. Never silently reinterpret a checkpoint from the other backend.
7. Satisfy the identity: `forward_sequence` and `forward` produce identical
   per-step `world_state` values in eval mode for the same inputs.

---

## 6. Every Current Code Path Affected by Replacing history/lengths with Recurrent State

| File | Lines | How it uses history/lengths | S0A change | S1 change |
|------|-------|---------------------------|------------|-----------|
| `rwm/models/rwm/model.py` | 113–174 | `forward()` builds `HistoryBuffer`, calls `world_hd(hist_seq, lengths=hist_len)`, returns `history`/`lengths` in output | Add `TemporalConfig.temporal_backend` branch | SRU branch ignores `history`, carries `z`; causal unchanged |
| `rwm/models/rwm/model.py` | 180–265 | `forward_sequence()` builds `all_tokens`, calls `world_hd(all_tokens, lengths, return_all=True)`, returns `history=all_tokens` | Add SRU scan branch; output `history` becomes placeholder | Remove HistoryBuffer construction in SRU forward |
| `rwm/imagination.py` | 112–147 | `warmup()` reads `out.history`/`out.lengths`, re-calls `world_hd` for per-step beliefs | Separate SRU warmup path that extracts beliefs from scan; causal unchanged | Remove re-entrant `world_hd` call |
| `rwm/imagination.py` | 175–220 | `advance()` uses `HistoryBuffer.from_history()`, calls `world_hd()` to get new belief | SRU advance keeps only `z_t`; causal unchanged | HistoryBuffer removed from advance |
| `rwm/imagination.py` | 226–285 | `rollout()` stores/manages `history`/`lengths` through steps | Separate SRU path; causal unchanged | HistoryBuffer removed from rollout |
| `rwm/evaluation/real_env_evaluator.py` | 190–191 | Stores `out.history`/`out.lengths` for next call to `model()` | Reads `out.world_state` instead for SRU | Conditional on backend |
| `rwm/evaluation/episode_evaluator.py` | 169–179 | Stores `out.history`/`out.lengths` for next call | Same change | Conditional on backend |
| `rwm/commands/rwm_manual_test.py` | 108–129 | Stores `output.history`/`output.lengths` for next call | Same change | Conditional on backend |
| `rwm/trainers/imagined_actor_critic.py` | 244–246 | Reads `warmup_state.history`/`warmup_state.lengths` | SRU warmup returns state; read `warmup_state.current_belief` | Remove history tracking |
| `rwm/utils/rollout_simulator.py` | 38–88, 91–131 | Full `HistoryBuffer` usage in warmup and imagine_rollout | Legacy code, not recommended for SRU migration; raise `NotImplementedError` with SRU migration message | Remove after S5 |
| `rwm/trainers/deterministic/world_model_trainer.py` | 203 | Tracks `gn_world_hd` for grad norms | Both backends report under `gn_temporal` | Rename logging key |
| `rwm/utils/history_buffer.py` | All | Entire class | No change for causal; SRU branch will not instantiate it | Mark as causal-only with deprecation warning in S6 |

### Compatibility-only code that should disappear after the final SRU decision

- `HistoryBuffer` — causal-only; kept for S0–S5, deprecated in S6 if SRU wins.
- `WorldModelOutput.history` and `WorldModelOutput.lengths` — causal-only fields
  in the NamedTuple; kept for S0–S5, deprecated in S6 if SRU wins.
- `CausalTransformer` module itself — removed in S6 if SRU wins.
- The `rollout_simulator.py` — legacy LSTM draft; should be removed regardless.

---

## 7. Config Extension

`TemporalConfig` gains the following fields.  Missing fields in old configs
default to `"causal_transformer"` for backward compatibility.

```python
@dataclasses.dataclass(frozen=True)
class TemporalConfig:
    """Causal Transformer or SRU temporal world model."""
    backend: str = "causal_transformer"   # "causal_transformer" | "minimal_sru"
    seq_len: int = 20
    world_state_dim: int = 80
    observational_dropout: float = 0.6
    warmup_steps: int = 5
    ffn_mult: int = 2
    transformer_dropout: float = 0.1
    # SRU-specific (ignored when backend == "causal_transformer"):
    sru_carry_bias_init: float = 1.0
    sru_burn_in_steps: int = 20
    sru_state_init: str = "loss_masked_burn_in"
```

`carry_bias_init` calibration (e.g., {1, 2, 3}) is an S1/S2 diagnostic, not
a new architecture ablation.

See `docs/plans/sru_temporal_validation_plan.md` for the full development
strategy and staged gates.
