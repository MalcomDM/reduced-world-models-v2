# Documentation Map and Current Execution Status

## Canonical Artifact Map

| Need | Canonical document | Role |
|---|---|---|
| Thesis direction and terminology | `Summary.md` | Concise research motivation; not an execution checklist. |
| Architecture, timing, objectives, and gradient boundaries | `technical_definitions.md` | Canonical design contract. |
| Rollout indexing and schema semantics | `contracts/transition_contract.md` | Canonical data contract; preserves historical bugs as labelled evidence. |
| Research hypotheses, evidence limits, performance audit, and ablations | `plans/architecture_validation_plan.md` | Canonical Stage 2.5 validation plan. |
| Implementation order and completion history | `plans/implementation_plan.md` | Canonical engineering stage plan. |
| Thesis-facing probe record | `evidence/theoretical_probes.md` | Concise claims, matched evidence, and limits. |
| Detailed experiment evidence | `evidence/` | Reproducible experiment reports. |
| Run/config/checkpoint conventions | `protocols/experiment_artifacts.md` | Reproducibility and artifact contract. |

When two documents appear to conflict, prefer this order:

1. `technical_definitions.md` for current architectural semantics;
2. `contracts/transition_contract.md` for dataset/indexing semantics;
3. `plans/architecture_validation_plan.md` for whether a claim is experimentally established;
4. `plans/implementation_plan.md` for execution sequence.

## Stage Map

| Status | Stage | Objective | Advancement gate |
|---|---|---|---|
| Complete | 0 | Freeze data/timing contracts | Contract and regression tests. |
| Complete | 0.5 | Reproducible experiment infrastructure | Structured runs, manifests, seeds, checkpoints. |
| Complete | 1 | Active causal Transformer migration | Structured temporal API and regression coverage. |
| Anchor complete | 2 | End-to-end factual reward prediction | Pipeline can overfit and exceeds the historical mean baseline. |
| Next | 2.5A | Measurement foundation | Versioned seeded scenarios, fixed splits, full-episode evaluator, controlled action branches. |
| Then | 2.5B | Correct and accelerate reward anchor | Correctness fixes preserve/match fixed-suite reward result. |
| Then | 2.5C | Perception hypotheses | Retained mechanisms beat declared matched baselines. |
| Then | 2.5D | Masked-dynamics hypothesis | Masked beliefs respond to branched actions and degrade gradually by horizon. |
| Then | 3 | Trainable imagination engine | One correct, differentiable observed/masked transition interface. |
| Then | 4 | Actor-Critic with frozen model | Policy/value mathematics and real evaluation work with stable beliefs. |
| Then | 5 | Actor-Critic through imagination | Imagined gains correlate with real return. |
| Then | 6 | Controlled upstream behavior gradients | Factual behavior pressure improves without predictive forgetting. |
| Then | 7 | Progressive collection/replay | Improvement persists across data-collection cycles and unseen seeds. |

## Immediate Work Boundary: Stage 2.5A

Stage 2.5A changes measurement and data provenance, not model learning. It must
deliver:

- a versioned rollout schema for newly collected data with seed, policy
  provenance, collector configuration, and separate `terminated`/`truncated`;
- immutable development, validation, and locked-test seed manifests;
- a full-episode evaluator that counts each transition once;
- a deterministic branch runner: replay a prefix on a fixed seed, branch into
  specified action sequences, and verify reproduced prefix frames/rewards;
- reusable attention instrumentation: score heatmap, hard Top-K overlay, and
  within-selected-patch pooling weights;
- regression tests for schema validation, deterministic replay, split isolation,
  unique-transition accounting, and visualization tensor alignment.

Existing Stage 2 data remains a historical anchor. New schema/data is not
silently mixed with it; manifests state which schema version was used.

## New Commands (Stage 2.5A)

```
rwm eval init-seeds <output> --dev-seeds ... --val-seeds ... --test-seeds ...
rwm eval collect <manifest> <seed> --out-dir ... [--operator ...]
rwm eval label <episode> --quality ... [--tags ...] [--operator ...] [--notes ...]
rwm eval status <eval-dir>
```

## Module overview (Stage 2.5A)

| Module | Purpose |
|--------|---------|
| `src/rwm/evaluation/schema.py` | Versioned rollout schema, seed manifest, episode metadata |
| `src/rwm/evaluation/collector.py` | Evaluation-only rollout collector |
| `src/rwm/evaluation/branch_runner.py` | Deterministic branch experiments |
| `src/rwm/evaluation/episode_evaluator.py` | Full-episode evaluation (unique transitions) |
| `src/rwm/evaluation/attention_trace.py` | Attention instrumentation and rendering |
| `docs/protocols/evaluation_protocol.md` | Detailed collection/labeling/evaluation protocol |
