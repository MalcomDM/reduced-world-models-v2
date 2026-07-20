# Documentation Map and Current Execution Status

## Canonical Artifact Map

| Need | Canonical document | Role |
|---|---|---|
| Thesis direction and terminology | `Summary.md` | Concise research motivation; not an execution checklist. |
| Architecture, timing, objectives, and gradient boundaries | `technical_definitions.md` | Canonical design contract. |
| Rollout indexing and schema semantics | `contracts/transition_contract.md` | Canonical data contract; preserves historical bugs as labelled evidence. |
| Factual memory and latent-cache semantics | `contracts/latent_memory_contract.md` | Canonical versioning, invalidation, gradient, sampling, and retention contract. |
| Research hypotheses, evidence limits, performance audit, and ablations | `plans/architecture_validation_plan.md` | Canonical Stage 2.5 validation plan. |
| Implementation order and completion history | `plans/implementation_plan.md` | Canonical engineering stage plan. |
| Temporal-backend decision | `plans/sru_temporal_validation_plan.md` | Completed MinimalSRU S0–S6 validation and adoption record. |
| Memory/replay validation | `plans/memory_replay_validation_plan.md` | Stage-7 hypotheses, invariants, matched controls, and advancement gates. |
| Thesis-facing probe record | `evidence/theoretical_probes.md` | Concise claims, matched evidence, and limits. |
| Detailed experiment evidence | `evidence/` | Reproducible experiment reports. |
| MinimalSRU evidence lineage | `evidence/sru_temporal/` | Tracked S2–S6 reports; raw binary artifacts stay under ignored `runs/`. |
| Joint-training evidence | `evidence/joint_training/` | Controlled shared-representation gates; raw checkpoints remain under ignored `runs/`. |
| Run/config/checkpoint conventions | `protocols/experiment_artifacts.md` | Reproducibility and artifact contract. |

When two documents appear to conflict, prefer this order:

1. `technical_definitions.md` for current architectural semantics;
2. `contracts/transition_contract.md` for dataset/indexing semantics;
3. `contracts/latent_memory_contract.md` for cached-state/replay semantics;
4. `plans/architecture_validation_plan.md` for whether a claim is experimentally established;
5. `plans/implementation_plan.md` for execution sequence.

## Stage Map

| Status | Stage | Objective | Advancement gate |
|---|---|---|---|
| Complete | 0 | Freeze data/timing contracts | Contract and regression tests. |
| Complete | 0.5 | Reproducible experiment infrastructure | Structured runs, manifests, seeds, checkpoints. |
| Complete | 1–2 | Factual reward-model pipeline | Held-out reward prediction and timing gates. |
| Complete | 2.5A–D | Measurement and component refinement | Evaluation isolation, corrected gradients, perception ablations, masked dynamics. |
| Complete | 3 | Trainable imagination engine | Correct observed/blind transition interface. |
| Complete | 4 | Actor-Critic mathematics | Bounded actions, λ-returns, target Critic, freeze contracts. |
| Complete | 5 | Frozen imagined Actor-Critic | Critic learns; real-environment evaluation works. |
| Complete | SRU S0–S6 | Temporal-backend replacement | MinimalSRU passes reward, blind-dynamics, z-only imagination, and policy-parity gates. |
| Complete (calibration deferred) | 6 | Safe ControllerTrunk + Critic boundary | Connectivity/predictive gates passed; the 500-update schedule needs gentler calibration after memory. |
| Active next | 7 | Factual memory, latent cache, and wake–dream replay | Probabilistic priority replay beats an equal-budget uniform baseline and establishes a stronger behavioral control. |
| Planned | 8 | Memory-informed upstream gradients and cycle calibration | Progressively open SRU/Actor/perception against the stronger Stage-7 baseline, then calibrate repeated cycles. |
| Planned | 9 | Thesis ablations and interpretation | Each retained component has matched evidence or is labelled unproven. |
| Planned | 10 | Final thesis evaluation | Multi-seed results under predeclared interaction and compute budgets. |

## Immediate Work Boundary: Stage 6

MinimalSRU is the selected backend on `main`; the causal baseline remains on
`baseline/causal-transformer-stage5`. Stage 6.0 established the gradient audit.
The final split-clean Stage-6.1 experiment preserved visible and masked
prediction in both seeds. Its specific 500-update schedule missed
real-return non-inferiority for seed 42, so it is not used during Stage 7.
This does not reject the gradient boundary: Stage 8 will calibrate smaller,
more frequent/interleaved updates against the stronger memory baseline.

Stage 7 makes factual experience memory a central part of the learning cycle.
Its optional versioned `z_t` cache accelerates frozen Actor-Critic
consolidation, but never replaces factual reconstruction when gradients must
enter the world model. MinimalSRU, Actor pressure and perception remain frozen
until memory establishes a stronger behavioral baseline; their progressive
opening is now Stage 8. The lifecycle and invalidation rules are specified in
`contracts/latent_memory_contract.md`.
