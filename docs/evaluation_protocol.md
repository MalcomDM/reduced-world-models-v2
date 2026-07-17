# Evaluation Protocol — Stage 2.5A

## Split Rules

Seeds are divided into three immutable groups:

| Split | Purpose | When used |
|-------|---------|-----------|
| `dev` | Implementation and tuning | During development, for debugging and configuration experiments |
| `val` | Architecture selection | For comparing model variants and selecting hyperparameters |
| `locked_test` | Final thesis results | Evaluated only for final thesis numbers; never during development |

**A seed belongs to exactly one split.** The split is recorded in the seed
manifest and cannot be changed after an episode is collected on that seed.
Seed manifests are validated on load: duplicate seeds and invalid split names
are rejected.

## Episode Labeling Conventions

Each collected episode has a `quality` field:

| Quality | Meaning |
|---------|---------|
| `unreviewed` | Default. Episode collected but not yet inspected. |
| `review` | Episode flagged for review (e.g., unusual reward pattern, suspected corruption). |
| `keep` | Episode is clean and usable for analysis. |
| `discard` | Episode should be excluded from analysis (e.g., operator error, corrupted data). |

Additional metadata stored with each episode:

- `scenario_tags`: comma-separated controlled tags (e.g., `"curve_left,high_speed"`)
- `operator`: name or ID of the person who collected/labeled the episode
- `notes`: free-text annotations

## Human Collection Design

Use one primary human trajectory per fixed track seed. Repeating a seed creates
correlated runs of the same road, so it is less valuable than adding another
seed. A practical initial benchmark contains **eight seeds per split**; collect
the first five development seeds as a pilot before committing the full set.

| Split | Initial collection | Use |
|---|---:|---|
| `dev` | 5 pilot, then 8 total seeds | UX checks, diagnostics, implementation work |
| `val` | 8 seeds, one primary trajectory each | architecture and hyperparameter selection |
| `locked_test` | 8 seeds, one primary trajectory each | final results only |

Target 700--1,000 transitions per kept trajectory; a clean partial trajectory
with at least 500 transitions is still usable. The first 50 transitions are a
startup stratum, not a reason to discard an otherwise good episode.

For each primary trajectory:

1. drive naturally and competently through the available road geometry;
2. include long centered forward-driving stretches whenever safe;
3. on selected development trajectories, include one 2--3 second coast and
   one conservative off-road/recovery event after a stable stretch;
4. do not repeatedly crash, stop, or oscillate merely to manufacture events.

Use only observed episode-level tags: `competent_drive`, `curve_left`,
`curve_right`, `coast`, `recovery`, `off_road`, and `low_speed`. Tags state
that an event occurred somewhere in the episode; they do not claim a labelled
frame interval. Use `review` for a questionable run and `discard` only for an
input/display failure or clearly unusable recording.

## Operator Workflow

### 1. Initialize seed manifest

```bash
rwm eval init-seeds data/eval/seeds.json \
    --dev-seeds 10,11,12 \
    --val-seeds 20,21 \
    --test-seeds 30,31,32
```

This creates a hash-provenanced manifest with three immutable groups. All
future collection reads splits from this file. It cannot be replaced after an
episode references its hash, even with `--force-replace`.

### 2. Collect episodes (headless)

```bash
rwm eval collect data/eval/seeds.json 10 \
  --policy-name human --render-mode human --fps 60 \
  --out-dir data/eval --operator "name"
```

The `eval-collect` command:
- loads the seed manifest and validates that seed `10` is registered;
- derives the split (`dev`) from the manifest — no user override is possible;
- creates `data/eval/dev/{episode_id}.npz` with separate `terminated`/`truncated`;
- creates `data/eval/dev/{episode_id}.episode.json` with metadata.

### 3. Label episodes (quality, tags, notes)

```bash
rwm eval label data/eval/dev/{episode_id}.npz \
    --quality keep \
    --tags "curve_left,high_speed" \
    --operator "name" \
    --notes "Clear left curve at high speed"
```

### 4. Check status

```bash
rwm eval status data/eval
```

Shows:
- number of episodes per split;
- quality breakdown (keep/review/discard/unreviewed);
- missing metadata files;
- manifest hash, path, seed, split, and purpose integrity checks.

## Forbidden Uses of Evaluation Data

- Evaluation episodes must NEVER appear in training datasets, data loaders,
  replay buffers, or checkpoint selection.
- The `purpose` field in every episode metadata is `evaluation_only` and must
  never be changed.
- Splits must never be overridden by user input during collection.
- Existing Stage 2 historical data (`data/rollouts/`) is never mixed with
  new schema evaluation data.

## READY FOR HUMAN COLLECTION

- [x] Seed manifest format and validation
- [x] Evaluation-only schema (`terminated`/`truncated` separate)
- [x] Training data isolation (different root path, different schema)
- [x] `eval-collect` command with split-from-manifest enforcement
- [x] `eval-label` command for quality/tags/operator/notes
- [x] `eval-status` command for episode counts and quality breakdown
- [x] Branch runner infrastructure (prefix replay, action branches, verification)
- [x] Full-episode evaluator (unique transitions, stratified metrics)
- [x] Attention instrumentation (trace API, heatmap, selected-patch overlay)
- [x] Human-mode lifecycle, paced input, and partial-save protection
- [x] Manifest integrity enforcement during metric evaluation

**Instructions for manual collection:**

With an X11 display available, run:

```bash
# 1. Create seed manifest
rwm eval init-seeds data/eval/seeds.json \
    --dev-seeds 10,11,12,13,14 \
    --val-seeds 20,21,22,23,24 \
    --test-seeds 30,31,32,33,34

# 2. Collect a dev episode. Close the window or press Escape to save cleanly.
rwm eval collect data/eval/seeds.json 10 \
  --policy-name human --render-mode human --fps 60 \
  --out-dir data/eval --operator "my_name"

# 3. Label each episode
rwm eval label data/eval/dev/{episode_id}.npz --quality keep --tags "..." --operator "..."

# 4. Verify status
rwm eval status data/eval

# 5. Run branch experiments (headless, no display needed)
python -c "
from rwm.evaluation.branch_runner import run_branch_experiment
import numpy as np
seed = 10
prefix = np.zeros((50, 3), dtype=np.float32)
prefix[:, 1] = 1.0  # full gas for 50 steps
branches = {
    'straight': np.array([[0.0, 1.0, 0.0]] * 20, dtype=np.float32),
    'left': np.array([[-1.0, 1.0, 0.0]] * 20, dtype=np.float32),
    'right': np.array([[1.0, 1.0, 0.0]] * 20, dtype=np.float32),
    'brake': np.array([[0.0, 0.0, 1.0]] * 20, dtype=np.float32),
}
exp = run_branch_experiment(seed, prefix, branches)
# Save results
from rwm.evaluation.branch_runner import save_branch_experiment
from pathlib import Path
save_branch_experiment(exp, Path('data/eval'))
print('Branch experiment saved')
"
```
