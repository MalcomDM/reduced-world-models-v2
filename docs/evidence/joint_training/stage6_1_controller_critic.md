# Stage 6.1 — ControllerTrunk + Critic: Split-Clean Two-Seed Evidence

## Experiment

Joint-training runs with split-clean AC prerequisites at
`runs/imagined_actor_critic/minimal_sru/04_joint_controller_critic_split_clean/`.

## Results

| Gate | Seed 42 | Seed 43 |
|------|:-------:|:-------:|
| Visible delta ≤ +0.02 | PASS (−0.0110) | PASS (−0.0161) |
| Masked H=1 delta ≤ +0.02 | PASS (−0.0074) | PASS (−0.0113) |
| Masked H=2 delta ≤ +0.02 | PASS (−0.0076) | PASS (−0.0146) |
| Masked H=4 delta ≤ +0.02 | PASS (−0.0090) | PASS (−0.0129) |
| Masked H=8 delta ≤ +0.02 | PASS (−0.0057) | PASS (−0.0080) |
| Masked H=12 delta ≤ +0.02 | PASS (−0.0034) | PASS (−0.0049) |
| Real return treatment ≥ control | FAIL (−82.35 vs −80.15) | PASS (identical −82.35) |
| Actor/SRU/perception unchanged | PASS | PASS |
| Zero train/val overlap | PASS | PASS |

## Interpretation

The visible and masked gates pass for both seeds, confirming that
ControllerTrunk joint training does not degrade factual prediction quality
beyond the ±0.02 non-inferiority threshold.

The real-return gate passes for seed 43 but fails for seed 42. The failure is
narrow: control mean −80.15 (driven by seed 101 at −76.90) vs treatment
−82.35 (seed 101 at −83.50). Seeds 100 and 102 are unchanged.

The seed-42 real-return regression could reflect the small three-track sample,
but the predeclared gate is not relaxed after observing it. The split-clean
frozen control is retained for Stage 7 so memory can be measured without a
second intervention.

This result does not reject the ControllerTrunk + Critic gradient boundary.
Five hundred consecutive updates can shift the shared representation away
from the frozen Actor even while reward prediction improves. Stage 8 will test
gentler schedules—fewer updates between validations, lower Controller/behavior
pressure, and interleaved factual anchoring—against the stronger
memory-trained policy.
