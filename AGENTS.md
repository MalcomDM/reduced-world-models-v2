# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives under `src/rwm/`. Keep models in `models/`, training logic in `trainers/` or `loops/`, Typer commands in `commands/`, and shared helpers in `datasets/`, `envs/`, and `utils/`. The `scripts/` tree contains standalone training, exploration, data-generation, and manual validation utilities. Tests are split between `tests/unit/` and `tests/integration/`. Generated rollouts, checkpoints, CSV logs, and `runs/` output should not be committed unless intentionally used as fixtures.

## Build, Test, and Development Commands

- `pip install -e .` installs the `rwm` package and CLI in editable mode using dependencies from `.devcontainer/requirements.txt`.
- `rwm --help` lists the Typer commands; use command-level help, such as `rwm collect --help`, before launching long jobs.
- `pytest` runs the complete test suite configured by `pytest.ini`.
- `pytest tests/unit` runs fast component tests.
- `pytest tests/integration -m integration` selects marked integration checks; some require CUDA, Gymnasium, rollout data, or display access.
- `python -m scripts.train.train_world_model --help` shows options for the standalone world-model trainer.

The `.devcontainer/` configuration provides Python 3.11+, PyTorch 2.7, CUDA 12.8, and NVIDIA runtime access. For interactive CarRacing rendering, run `xhost +local:root` on the host as described in `README.md`.

## Coding Style & Naming Conventions

Use standard Python conventions: four-space indentation, `snake_case` for functions/modules, `PascalCase` for classes, and `UPPER_CASE` for constants. Add type annotations to public APIs and keep tensor shapes/device assumptions explicit. Prefer small modules and reusable helpers over adding logic to CLI commands. No formatter or linter is currently configured; keep imports grouped and match nearby style without introducing tabs.

## Testing Guidelines

Tests use pytest. Name files `test_*.py`, tests `test_*`, and reusable setup as fixtures. Place isolated behavior in `tests/unit/`; reserve integration tests for CLI, training-loop, CUDA, environment, or filesystem interactions. Apply markers from `pytest.ini` (for example `models`, `training`, `dataset`, or `memory`). Add regression coverage with every bug fix. There is no enforced coverage percentage, so prioritize assertions around outputs, shapes, state changes, and errors.

## Commit & Pull Request Guidelines

History uses short, imperative summaries such as `Added behavior memory` or `Changing to v2`. Keep commits focused and describe the outcome in the subject; add rationale in the body when behavior or model architecture changes. Pull requests should include a concise problem/solution summary, commands and tests run, related issue links, and any CUDA/data prerequisites. Attach plots, logs, or screenshots when training behavior or rendered output changes, and avoid committing large datasets or checkpoints.
