# SRU Temporal — Run Index

**Lineage status:** COMPLETE. MinimalSRU adopted on `main`; causal reference
preserved on `baseline/causal-transformer-stage5`.

| Path | Seeds | Config | Best val MSE | Best ratio | Probe | Date |
|------|:-----:|--------|:------------:|:----------:|:-----:|------|
| `01_visible_reward_anchor/` | 42, 43 | S2 SRU RBi B=20 | 0.481/0.496 | 0.854/0.791 | 4/4 | 2026-07-19 |
| `02_macroblock_m64_matched_exposure/` | 42, 43 | S2.5 Macroblock M=64 | 0.491/0.537 | 0.872/0.854 | 4/4 | 2026-07-19 |
| `03_matched_backend_evaluation/` | 42, 43 | Frozen C/SRU comparison | — | 0.840/0.761 | 4/4 | 2026-07-19 |
| `04_observational_dropout_anchor/` | 42, 43 | S3 Post-perc mask B=20 | 0.471/0.487 | 0.698/0.611 | 4/4 | 2026-07-19 |
| `05_canonical_burnin_comparison/` | 42, 43 | S4B Causal/SRU-20/8/4/0 | 0.468/0.400 | 0.831/0.637 | 4/4 | 2026-07-19 |
| `06_sru4_matched_wallclock/` | 42, 43 | S4B SRU-4 17ep | — | 0.871 | 4/4 | 2026-07-19 |
| `07_cold_start_36/` | 42, 43 | S4C CS-36 B=0 seq=36 | 0.479/0.550 | 0.903/0.906 | 4/4 | 2026-07-19 |
| `08_strict_observational_dropout_anchor/` | 42, 43 | S4C.1 Strict pre-perc mask | 0.500/0.512 | 0.887/0.814 | 4/4 | 2026-07-19 |
| `09_retained_components/` | 42, 43 | S4C learned/fixed K8, learned K16 | 0.448–0.512 | visible 0.755–0.851; masked 0.937–0.966 | 4/4 | 2026-07-19 |

Final frozen-control parity report: [10_frozen_actor_critic_parity.md](10_frozen_actor_critic_parity.md).
