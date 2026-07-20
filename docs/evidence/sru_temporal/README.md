# SRU temporal lineage

Artifacts for the dynamic-latent-state research line based on true
observational dropout and a compact SRU-like recurrent temporal state.

The tracked validation plan and decision gates are in
[sru_temporal_validation_plan.md](../../plans/sru_temporal_validation_plan.md).

Frozen causal code baseline:

- branch: `baseline/causal-transformer-stage5`
- commit: `a0406262c6b3b2db647a7675e4bd6125e157b16f`

**Decision:** MinimalSRU was adopted as the primary temporal backend on
2026-07-20 after completing S0–S6. The causal branch remains the reproducible
comparison baseline.

The numbered reports here are immutable tracked experiment evidence. See
[run_index.md](run_index.md) for the compact index. Raw checkpoints, metrics,
JSON, CSV, and plots remain under ignored `runs/` paths recorded by each
report.
