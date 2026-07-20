# Script layout

Active scripts are grouped by purpose:

- `data/`: reproducible data-preparation utilities.
- `evaluation/`: reusable checkpoint, reward, masked-dynamics, and real-environment evaluators.
- `training/`: standalone training entry points.
  `train_joint_controller_critic.py` is the bounded Stage 6.1 shared-
  representation gate.
- `benchmarks/`: throughput and execution-policy measurements.
- `profiling/`: data-pipeline profiling.
- `experiments/sru/`: narrowly scoped, reproducible SRU comparison runners.
- `legacy/`: historical `app.*` scripts retained only as reference; they are not supported entry points.

Run scripts from the repository root, for example:

```bash
python scripts/evaluation/evaluate_reward_prediction.py --help
python scripts/data/build_frame_cache.py --help
```
