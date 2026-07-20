# S4B — Canonical Visible Temporal / Burn-In Comparison

**Status:** complete. All anchors use the same cache, seeds, K=8, beta=0.1,
linear head, posterior-mean evaluation, and 10 dataset epochs. Observational
dropout is disabled.

## Corrected frozen-checkpoint evaluation

The evaluator reconstructs each checkpoint's configured SRU burn-in and scores
only target positions. Ten earlier outputs that omitted SRU context were deleted
and replaced; no anchors were retrained.

| Variant | Seed | Frozen MSE | Frozen ratio | Time | GPU |
|---|---:|---:|---:|---:|---:|
| Causal | 42 | 0.4683 | 0.831 | 153 s | 0.56 GB |
| Causal | 43 | 0.5155 | 0.819 | 164 s | 0.56 GB |
| SRU-20 | 42 | 0.4718 | 0.837 | 374 s | 1.23 GB |
| SRU-20 | 43 | 0.3999 | 0.635 | 399 s | 1.23 GB |
| SRU-8 | 42 | 0.4822 | 0.856 | 266 s | 0.83 GB |
| SRU-8 | 43 | 0.5219 | 0.829 | 278 s | 0.83 GB |
| SRU-4 | 42 | 0.5036 | 0.894 | 228 s | 0.69 GB |
| SRU-4 | 43 | 0.4970 | 0.790 | 239 s | 0.69 GB |
| SRU-0 | 42 | 0.4841 | 0.859 | 190 s | 0.56 GB |
| SRU-0 | 43 | 0.5279 | 0.839 | 203 s | 0.56 GB |

| Variant | Mean frozen ratio | Mean time | Mean GPU |
|---|---:|---:|---:|
| Causal | 0.825 | 159 s | 0.56 GB |
| SRU-20 | **0.736** | 387 s | 1.23 GB |
| SRU-8 | 0.843 | 272 s | 0.83 GB |
| SRU-4 | 0.842 | 234 s | 0.69 GB |
| SRU-0 | 0.849 | 197 s | 0.56 GB |

## Gate

The predeclared SRU-20 tolerance is `0.02`: threshold `0.756`.

- All variants beat their constant-mean baseline in both seeds.
- SRU-8, SRU-4, and SRU-0 are respectively `+0.106`, `+0.105`, and `+0.113`
  ratio points worse than SRU-20.
- **Selected context: SRU burn-in 20.**

The seed spread for SRU-20 (`0.837` vs `0.635`) is material. This selects the
context for the next strict-dropout test; it does not select SRU over Causal.
