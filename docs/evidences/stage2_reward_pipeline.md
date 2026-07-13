# Stage 2 — Reward Pipeline Validation

## Purpose

Produce valid evidence about whether the corrected Stage 2 reward model
can learn from real rollout data, before advancing to Stage 3.

## Timing Contract (approved)

```
belief b_t = Transformer(obs[t], action[t-1], history)
RewardHead(shared(belief_t), action[t]) → reward[t] (= r_{t+1})
```

## Metric definitions

| Metric | Definition |
|--------|------------|
| ``train_mse`` | Mean ``F.mse_loss(pred_reward, true_reward)`` over all batches in the epoch |
| ``train_kl`` | Mean ``kl_normal(tok_mu, tok_logvar)`` over all batches (unweighted, logged for diagnostic) |
| ``train_total`` | ``train_mse + beta * train_kl`` |
| ``val_mse`` | ``sum((pred - true)^2) / N`` over all held-out windows |
| ``val_mae`` | ``sum(|pred - true|) / N`` |
| ``baseline_mse`` | ``sum((true - train_mean)^2) / N`` where ``train_mean`` is the actual training-set reward mean from the most recent epoch |

## Dataset

| Property | Value |
|----------|-------|
| Root | ``data/rollouts/rwm_deterministic/scenario_0`` |
| Total files | 20 (10 random, 10 random_smooth) |
| Total steps | 5339 |
| Episode length range | 148–845 |
| Reward mean / std | 0.078 / 0.830 |
| No done flags | All idle-stopped |

## Experiments

### Two-seed comparison (beta=0, reward-MSE only)

| Metric | Seed 42 | Seed 43 |
|--------|---------|---------|
| Train files / Val files | 16 / 4 | 16 / 4 |
| Train windows | 4040 | 4304 |
| Val windows (held-out) | 256 | 256 |
| Batch size / Sequence len | 8 / 16 | 8 / 16 |
| Best val MSE | **0.4720** | **0.5390** |
| Baseline MSE | 0.5634 | 0.6295 |
| **Model/baseline ratio** | **0.838** ✅ | **0.856** ✅ |
| Epochs / Time | 10 / 500 s | 10 / 536 s |
| Peak GPU | 0.27 GB | 0.27 GB |
| Checkpoint | ``runs/stage2-val/seed42_beta0.0/checkpoint_best.pt`` | ``runs/stage2-val/seed43_beta0.0/checkpoint_best.pt`` |

**Command (seed 42):**
```bash
python scripts/validate_stage2.py --beta 0.0 --epochs 10 \
    --out runs/expA --max-val-windows 256 --batch-size 8 --seed 42
```

**Command (seed 43):**
```bash
python scripts/validate_stage2.py --beta 0.0 --epochs 10 \
    --out runs/stage2_confirmation_seed43 --max-val-windows 256 --batch-size 8 --seed 43
```

### Experiment B: beta=1.0 (normal KL weight)

**Command:**
```bash
python scripts/validate_stage2.py --beta 1.0 --epochs 10 \
    --out runs/expB --max-val-windows 256 --batch-size 8 --seed 42
```

**Run directory:** ``runs/stage2-val/seed42_beta1.0/``

| Metric | Value |
|--------|-------|
| Best val MSE | **0.5544** |
| Baseline MSE | 0.5634 |
| **Model/baseline ratio** | **0.984** (beats baseline by 1.6%) |
| Training time | 496 s |

### Action probe (trained checkpoint)

**Checkpoint:** ``runs/stage2-val/seed42_beta0.0/checkpoint_best.pt``

| Action | Reward |
|--------|--------|
| all zeros | 0.2089 |
| full steer | 0.1762 |
| full gas | 0.2153 |
| full brake | 0.1851 |

All 4 actions produce unique predictions — the reward head is properly
current-action-conditioned after training.

## Interactive human-driving diagnostic

**Checkpoint:** ``runs/stage2-val/seed43_beta0.0/checkpoint_best.pt``

The restored ``rwm test-rwm-manually`` command was used to drive one
CarRacing episode interactively while plotting immediate predicted and real
reward.  The command persisted ``756`` aligned transition pairs to
``runs/manual_reward_eval.csv``.

| Metric | Value |
|--------|-------|
| True-reward mean | 0.4273 |
| Predicted-reward mean | 0.3417 |
| MSE / MAE | 1.6021 / 0.7307 |
| Pearson correlation | 0.2551 |
| Mean prediction, positive-reward steps | 0.7252 |
| Mean prediction, non-positive steps | 0.2744 |

The model assigns roughly 2.6× higher predicted reward to the positive
road-progress steps than to the ``-0.1`` non-progress steps.  This is useful
qualitative evidence that it distinguishes the reward context during human
driving.  It is not a replacement for held-out validation: the episode is a
single human-policy trajectory, and reward magnitude remains poorly
calibrated.

## Conclusions

### Pipeline status: PASSED

The entire reward-prediction pipeline is functional:
- Perception → Transformer → ControllerTrunk → reward head
- Timing contract is correct and regression-tested
- Full-sequence and incremental inference match
- Per-frame perception runs once; Transformer runs once per sequence
- Gradient flow reaches all blocks (no NaN, finite norms)

### Two-seed reproducibility: PASSED

Both independent seeds (42 and 43) produce a model that beats the held-out
constant-mean baseline:

| Seed | Model/baseline ratio | Margin |
|------|---------------------|--------|
| 42 | 0.838 | 16.2% below baseline |
| 43 | 0.856 | 14.4% below baseline |

The result is reproducible across independent train/validation splits.

### Data status: adequate

The 20-file dataset with episode-safe split produces consistent results
across two seeds.  Each seed uses a different file-level partition (same
ratio, different files), confirming that the signal is not specific to one
split configuration.

### Runtime profile

| Phase | Throughput |
|-------|-----------|
| Training (batch size 8, seq len 16) | ~11 batches/s |
| Validation (batch size 8, seq len 16) | ~6.5 batches/s |
| Time per epoch | ~50 s |
| Peak GPU memory | 0.27 GB |

### Stage 3 authorization: AUTHORIZED

Stage 2 reward validation **passes** with two independent seeds both
beating the baseline by 14-16%.  The reward-prediction pipeline is
functional, reproducible, and ready for Stage 3 (open-loop imagination).

**Conditions:**
- Use ``beta=0`` or a reduced ``beta`` (e.g., 0.1) for reward-focused
  training.
- Validation should continue to use the existing episode-safe split and
  bounded held-out protocol.
