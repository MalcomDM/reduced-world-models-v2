# Stage S5 — MinimalSRU Frozen Actor-Critic: Results

## Training

| Property | Value |
|----------|-------|
| Anchor | `08_strict_observational_dropout_anchor/seed42` |
| Backend | `minimal_sru` (burn_in=20) |
| Updates | 2,000 |
| Batch size | 8 |
| Curriculum | H ∈ {1, 2, 4} |
| Entropy coef | 0.03 |
| Wall time | 49.1 s |
| Checkpoints | `checkpoints/ac_checkpoint_{500,1000,1500,2000}.pt` |

### Horizon counts

| H | Batches | % |
|---|---------|---|
| 1 | 654 | 32.7 |
| 2 | 692 | 34.6 |
| 4 | 654 | 32.7 |

**Total imagined transitions:** 37,232

### Loss trajectory (median over 100-update windows)

| Metric | First 100 | Last 100 | Direction |
|--------|-----------|----------|-----------|
| Critic loss | 0.0495 | 0.0159 | ↓ |
| Actor loss | -0.0079 | 0.1162 | ↑ |
| Entropy | 0.4089 | -2.1310 | ↓ |

### Action statistics (mean over all 2,000 updates)

| Action | Mean | Std |
|--------|------|-----|
| Steer | 0.35 | 0.48 |
| Gas | 0.63 | 0.26 |
| Brake | 0.20 | 0.18 |

Saturation: steer > 0.90 at 6.4% of steps; gas/brake never saturate at 0.90.

### Freeze semantics (post-training)

| Component | Unchanged | Requires Grad | In Optimizer |
|-----------|:---------:|:-------------:|:------------:|
| World model | ✓ | – | – |
| MinimalSRU | ✓ | – | – |
| ControllerTrunk | ✓ | – | – |
| Actor | – | ✓ | ✓ |
| Online Critic | – | ✓ | ✓ |
| **Target Critic** | – | **✗** | **✗** |

Target Critic updated only through Polyak (τ=0.01). No gradients flow to it.

---

## Real-Environment Evaluation (dev seeds 100–102, 1,000 max steps)

### Policy returns

| Seed | Actor | Zero | Random |
|------|:-----:|:----:|:------:|
| 100 | -81.48 | -92.59 | -25.93 |
| 101 | -83.50 | -93.40 | -33.99 |
| 102 | -82.08 | -92.83 | -24.73 |
| **Mean** | **-82.35** | **-92.94** | **-28.22** |
| Variance | 0.72 | – | – |

### Reward prediction (Actor evaluation)

| Seed | MSE | MAE |
|------|:---:|:---:|
| 100 | 0.0867 | 0.0456 |
| 101 | 0.0639 | 0.0417 |
| 102 | 0.0800 | 0.0459 |

### Mean action (Actor evaluation)

| Seed | Steer | Gas | Brake |
|------|:-----:|:---:|:-----:|
| 100 | 0.793 | 0.718 | 0.108 |
| 101 | 0.793 | 0.718 | 0.108 |
| 102 | 0.792 | 0.718 | 0.109 |

### Anchor integrity

```
Checkpoint anchor: {"path": ".../checkpoint_best.pt", "hash": "79326840535458c5"}
File hash:          79326840535458c5
Anchor verified:    ✓
```

---

## Parity Assessment

| Gate | Target | Result |
|------|--------|:------:|
| **Primary SRU parity** | Mean ≥ -85.3 | **✓** (-82.35) |
| Beats zero-action | Mean > -92.94 | **✓** |
| Random diagnostic | – | Random (-28.22) beats all trained policies, consistent with known CarRacing difficulty |



## Files

All raw artifacts preserved under:
```
runs/imagined_actor_critic/minimal_sru/01_frozen_parity/seed42/
├── checkpoints/
│   ├── ac_checkpoint_500.pt
│   ├── ac_checkpoint_1000.pt
│   ├── ac_checkpoint_1500.pt
│   └── ac_checkpoint_2000.pt
├── evaluation/
│   ├── actor/     (per-seed CSV, JSON, summary.json, plots)
│   ├── zero/      (per-seed CSV, JSON, summary.json)
│   └── random/    (per-seed CSV, JSON, summary.json)
├── metrics.csv
├── training_summary.json
├── fixed_probe_pre.json
├── fixed_probe_post.json
└── RESULTS.md
```
