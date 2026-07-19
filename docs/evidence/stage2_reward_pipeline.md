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
| Best val MSE | **0.4650** | **0.5390** |
| Baseline MSE | 0.5634 | 0.6295 |
| **Model/baseline ratio** | **0.825** ✅ | **0.856** ✅ |
| Epochs / Time | 10 / 461 s | 10 / 536 s |
| Peak GPU | 0.27 GB | 0.27 GB |
| Checkpoint | ``runs/component_refinement/causal_transformer/00_reward_anchor_pre_kl_fix/seed_42/checkpoint_best.pt`` | ``runs/component_refinement/causal_transformer/00_reward_anchor_pre_kl_fix/seed_43/checkpoint_best.pt`` |

**Command (seed 42):**
```bash
python scripts/evaluate_reward_prediction.py --beta 0.0 --epochs 10 \
    --out runs/component_refinement/causal_transformer/00_reward_anchor_pre_kl_fix/seed_42/reproduction_seed42 --max-val-windows 256 --batch-size 8 --seed 42
```

**Command (seed 43):**
```bash
python scripts/evaluate_reward_prediction.py --beta 0.0 --epochs 10 \
    --out runs/component_refinement/causal_transformer/00_reward_anchor_pre_kl_fix/seed_43 --max-val-windows 256 --batch-size 8 --seed 43
```

### Retired pre-fix beta=1.0 diagnostic

The former beta=1.0 diagnostic reached a 0.984 model/baseline ratio. It is
recorded here only to explain why KL comparisons were deferred: KL was then
reduced incorrectly, so its checkpoint and transient artifacts were removed.

| Metric | Value |
|--------|-------|
| Best val MSE | **0.5544** |
| Baseline MSE | 0.5634 |
| **Model/baseline ratio** | **0.984** (beats baseline by 1.6%) |
| Training time | 496 s |

### Action probe (trained checkpoint)

**Checkpoint:** ``runs/component_refinement/causal_transformer/00_reward_anchor_pre_kl_fix/seed_42/checkpoint_best.pt``

| Action | Reward |
|--------|--------|
| all zeros | 0.1201 |
| full steer | 0.0901 |
| full gas | 0.1176 |
| full brake | 0.0967 |

All 4 actions produce unique predictions — the reward head is properly
connected to the current action after training. This is a structural probe,
not evidence that the model learned a useful state-dependent action effect: the
head is linear over `[shared_belief, action]`, so any nonzero action weight
creates unique values.

### Post-validation action audit

The 5,339 collected transitions contain no exact all-zero action rows and 4,928
distinct action vectors after rounding to three decimals. The action ablation
was therefore not accidentally performed on an all-zero dataset.

On the seed-42 bounded validation set, however, normal MSE (`0.46899`) and
shuffled-current-action MSE (`0.46917`) are effectively identical. The valid
conclusion is narrow: the direct current-action reward branch contributes
almost nothing on this sparse random-policy distribution. It does not test the
importance of previous actions in temporal context or multi-step action effects
under a competent policy. A controlled masked-horizon action-branch test is
defined in `docs/plans/architecture_validation_plan.md`.

## Interactive human-driving diagnostic

**Checkpoint:** ``runs/component_refinement/causal_transformer/00_reward_anchor_pre_kl_fix/seed_43/checkpoint_best.pt``

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

**Artifact warning (2026-07-17 audit):** the CSV path was later overwritten and
now contains 1,045 rows from two unequal episodes, not the 756 rows summarized
above. It stores neither actions, frames, nor environment seeds. The table is a
historical observation whose original raw artifact no longer survives; it must
not be cited as a reproducible thesis result. The qualitative “centered but
stopped” manual probe is useful evidence against a pure static-road shortcut,
but it cannot distinguish action history, visual motion, or HUD/speed cues.

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
| 42 | 0.825 | 17.5% below baseline |
| 43 | 0.856 | 14.4% below baseline |

The result is reproducible across independent train/validation splits.

### Data status: adequate only for a pipeline anchor

The 20-file dataset with episode-safe split produces a repeatable initial
signal across two seeds. Each seed uses a different file-level partition. The
result is enough to retain this checkpoint as an end-to-end anchor, but it is
not yet adequate for claims about action-conditioned dynamics or architectural
superiority: 95.24 percent of rewards are `-0.1`, every episode starts with the
same 20 full-gas actions, and stride-one validation windows repeatedly count
the same transitions.

### Runtime profile

| Phase | Throughput |
|-------|-----------|
| Training (batch size 8, seq len 16) | ~11 batches/s |
| Validation (batch size 8, seq len 16) | ~6.5 batches/s |
| Time per epoch | ~50 s |
| Peak GPU memory | 0.27 GB |

### Stage 3 authorization: WITHDRAWN PENDING STAGE 2.5

Stage 2 reward validation passes as a **functional pipeline checkpoint** with
two independent seeds beating the constant-mean baseline by 14--16 percent.
The post-validation audit found stronger confounds and several inactive or
incorrect research mechanisms: disconnected Top-K surrogate gradients,
incorrect temporal KL aggregation, false action resets at mid-episode window
starts, a state-independent linear action effect, and observational dropout
being intentionally disabled.

Proceed first through the fixed scientific suite, correctness repairs,
perception ablations, and masked-dynamics gate in
`docs/plans/architecture_validation_plan.md`. Use `beta=0` only to reproduce the
historical anchor; nonzero-beta conclusions require the KL fix.
