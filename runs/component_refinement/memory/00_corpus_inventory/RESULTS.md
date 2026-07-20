# 00 Corpus Inventory — RESULTS

Date: (generated)

Data root: `data/rollouts/rwm_deterministic/scenario_0/`

Seeds: [42, 43]

Horizons: [1, 2, 4, 8, 12]

d-horizons: [2, 3, 4]

---

## seed42

- Files: 16 train, 4 val
- Transitions: 4280
- Eligible pointers: 4280
- Disjoint train/val: True
- Ep lengths: {'min': 148.0, 'q1': 156.75, 'median': 218.0, 'q3': 270.75, 'max': 845.0, 'mean': 267.5, 'std': 170.760432}
- Immediate reward: {'min': -0.1, 'q1': -0.1, 'median': -0.1, 'q3': -0.1, 'max': 10.809091, 'mean': 0.078157, 'std': 0.838919}

### Return quantiles

- H=1: {'min': -0.1, 'q1': -0.1, 'median': -0.1, 'q3': -0.1, 'max': 10.809091, 'mean': 0.078157, 'std': 0.838919}
- H=12: {'min': -1.2, 'q1': -1.2, 'median': -1.2, 'q3': 2.436364, 'max': 16.785612, 'mean': 0.717484, 'std': 3.089024}
- H=2: {'min': -0.2, 'q1': -0.2, 'median': -0.2, 'q3': -0.2, 'max': 10.709091, 'mean': 0.13057, 'std': 1.083531}
- H=4: {'min': -0.4, 'q1': -0.4, 'median': -0.4, 'q3': -0.4, 'max': 10.509091, 'mean': 0.238854, 'std': 1.443592}
- H=8: {'min': -0.8, 'q1': -0.8, 'median': -0.8, 'q3': 2.374603, 'max': 13.588489, 'mean': 0.469622, 'std': 2.236058}

### Surprise counts

- h=2: {'up_count': 423, 'up_pct': 9.88, 'down_count': 441, 'down_pct': 10.3}
- h=3: {'up_count': 553, 'up_pct': 12.92, 'down_count': 556, 'down_pct': 12.99}
- h=4: {'up_count': 625, 'up_pct': 14.6, 'down_count': 644, 'down_pct': 15.05}

### Sensitivity grid (ESS)

| Config | ESS | ESS ratio |
|--------|-----|-----------|
| uniform | 4280.0 | 1.0 |
| return_only | 2384.0 | 0.557 |
| return_sharp | 1616.9 | 0.3778 |
| change_focused | 1643.5 | 0.384 |
| balanced | 2079.1 | 0.4858 |
| high_floor | 3778.3 | 0.8828 |
| return_extreme | 1636.3 | 0.3823 |

### Dense-region impact

- {'top_10pct_weight_fraction': 0.0393, 'top_25pct_weight_fraction': 0.0966, 'top_50pct_weight_fraction': 0.1904, 'gini_weight': 0.5007}

---

## seed43

- Files: 16 train, 4 val
- Transitions: 4546
- Eligible pointers: 4546
- Disjoint train/val: True
- Ep lengths: {'min': 148.0, 'q1': 166.5, 'median': 208.5, 'q3': 341.5, 'max': 845.0, 'mean': 284.125, 'std': 182.29539}
- Immediate reward: {'min': -0.1, 'q1': -0.1, 'median': -0.1, 'q3': -0.1, 'max': 10.809091, 'mean': 0.07234, 'std': 0.817673}

### Return quantiles

- H=1: {'min': -0.1, 'q1': -0.1, 'median': -0.1, 'q3': -0.1, 'max': 10.809091, 'mean': 0.07234, 'std': 0.817673}
- H=12: {'min': -1.2, 'q1': -1.2, 'median': -1.2, 'q3': 2.155705, 'max': 16.785612, 'mean': 0.662754, 'std': 3.043405}
- H=2: {'min': -0.2, 'q1': -0.2, 'median': -0.2, 'q3': -0.2, 'max': 10.709091, 'mean': 0.120799, 'std': 1.059255}
- H=4: {'min': -0.4, 'q1': -0.4, 'median': -0.4, 'q3': -0.4, 'max': 10.509091, 'mean': 0.220885, 'std': 1.412195}
- H=8: {'min': -0.8, 'q1': -0.8, 'median': -0.8, 'q3': 2.315265, 'max': 13.588489, 'mean': 0.434051, 'std': 2.203693}

### Surprise counts

- h=2: {'up_count': 442, 'up_pct': 9.72, 'down_count': 488, 'down_pct': 10.73}
- h=3: {'up_count': 574, 'up_pct': 12.63, 'down_count': 617, 'down_pct': 13.57}
- h=4: {'up_count': 640, 'up_pct': 14.08, 'down_count': 715, 'down_pct': 15.73}

### Sensitivity grid (ESS)

| Config | ESS | ESS ratio |
|--------|-----|-----------|
| uniform | 4546.0 | 1.0 |
| return_only | 2517.2 | 0.5537 |
| return_sharp | 1712.8 | 0.3768 |
| change_focused | 1727.7 | 0.3801 |
| balanced | 2189.2 | 0.4816 |
| high_floor | 4005.4 | 0.8811 |
| return_extreme | 1723.7 | 0.3792 |

### Dense-region impact

- {'top_10pct_weight_fraction': 0.0359, 'top_25pct_weight_fraction': 0.0946, 'top_50pct_weight_fraction': 0.1954, 'gini_weight': 0.5034}

---


