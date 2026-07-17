# Thesis-Relevant Probe Index

## Purpose

Short index of probes that may support the final thesis. Detailed commands,
metrics, and artifacts remain in the linked reports and run directories.

| Probe | What was tested | Scope and current result | Detailed record |
|---|---|---|---|
| End-to-end reward anchor | Complete perception → temporal representation → reward path versus a constant mean-reward baseline. | Two held-out file splits, seeds 42/43. The model beats baseline on both. This is reward-prediction evidence, not control evidence. | [Stage-2 reward pipeline](stage2_reward_pipeline.md); `runs/component_refinement/02_vectorized_reward_anchor/` |
| Local reward-rate diagnostic | Whether predictions follow sustained driving progress despite sparse CarRacing reward pulses. | One human-driven qualitative trace; target uses a causal three-step reward average. Indicates progress tracking but optimistic no-progress predictions. Not a benchmark. | [Notebook 1](../../notebooks/01_reward_prediction_evidence.ipynb) |
| Nonlinear reward head | Linear reward readout versus `83 → 32 → 1` ReLU head. | Held-out reward prediction, seeds 42/43. Linear wins on both; nonlinear remains an optional unselected variant. This does not determine future Critic/Actor head capacity. | `runs/component_refinement/03_nonlinear_reward_head/` |
| Adaptive Top-K selection | Learned K=8 patch candidates versus fixed-uniform and fixed-random K=8 candidate sets. | Held-out reward prediction, seeds 42/43. Learned selection wins against both static controls. Supports adaptive candidate selection, not semantic recognition, optimal K, or full inference savings. | `runs/component_refinement/04_topk_selection_ablation/`; [run index](../../runs/component_refinement/RUN_INDEX.md) |

## Pending thesis-relevant probes

- Learned K-capacity curve: K=4, 16, 32.
- Stochastic tokenizer under corrected KL weighting.
- Observational dropout / masked dynamics with controlled branches.
- Critic and Actor contributions after imagined-transition semantics are validated.
