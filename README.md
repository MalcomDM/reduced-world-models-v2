# Reduced World Models

Research code for a thesis exploring whether an agent can learn useful environment dynamics through a deliberately constrained internal representation. The current environment is `CarRacing-v3` from Gymnasium.

The central hypothesis is that a world model does not need to preserve every visual detail. Instead, aggressive spatial bottlenecks and observational dropout may encourage it to retain only information that matters for predicting reward and choosing actions. This repository is an experimental implementation of that idea, not yet a stable training framework.

## Research Objective

The model should learn a compact world state from observations, actions, and rewards, then use that state to train a controller through imagined trajectories. The long-term objective is an end-to-end trainable system where reward-driven gradients can influence the complete path back to the visual encoder and attention mechanism.

Earlier iterations trained the world model first and the controller afterward. This makes the implementation simpler, but creates a moving-representation problem: improving one component may invalidate the representations or behaviors learned by another, and repeated training can cause catastrophic forgetting. The current design investigates whether selective perception and strong dropout can produce a smaller, more stable representation, but this remains an open theoretical and experimental question.

## Current Architecture

```text
RGB observation
    │
    ▼
CNN encoder
    │
    ▼
Overlapping stochastic patch tokens + positional encoding
    │
    ▼
Attention scoring → differentiable Top-K selection
    │
    ▼
Weighted spatial representation
    │
    ▼
Observational dropout + action history
    │
    ▼
Causal Transformer → world state + predicted reward
    │
    ▼
Controller → steering, gas, brake
    │
    ▼
Imagined rollouts → behavior selection and memory
```

For a `64×64` frame, the encoder produces a `16×32×32` feature map. It is divided into 225 overlapping patch tokens, of which the attention mechanism selects eight. These are pooled into a 32-dimensional spatial representation. After concatenating actions, a causal Transformer produces an 80-dimensional world state and a reward estimate.

The implementation is currently being migrated from an LSTM temporal state to Transformer history. Some callers still use the old interface; see [PENDING_WORK.md](PENDING_WORK.md).

Container rendering, GPU monitoring, and legacy experiment observations are recorded in [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md).

## Repository Structure

- `src/rwm/data/`: rollout collection and the active windowed dataset.
- `src/rwm/models/rwm/`: encoder, tokenization, selection, spatial pooling, dropout, and temporal model.
- `src/rwm/models/controller/`: policy network mapping world states to continuous actions.
- `src/rwm/trainers/` and `src/rwm/loops/`: world-model and controller training orchestration.
- `src/rwm/utils/`: imagined rollouts, behavior memory, preprocessing, and callbacks.
- `src/rwm/commands/`: Typer command implementations exposed by the `rwm` CLI.
- `tests/unit/`: isolated model, policy, collector, and dataset tests.
- `tests/integration/`: CUDA, Gymnasium, training, and memory workflows.
- `scripts/`: older experiments and manual utilities; several still reference the previous `app.*` package layout.

Generated datasets and experiment runs belong under `data/` and `runs/`; both are ignored by Git.

## Rollout Format

Each collected episode is stored as a compressed `.npz` file:

```text
obs     (T, H, W, 3)  uint8
action  (T, 3)        float32
reward  (T,)          float
done    (T,)          bool
```

A neighboring `.info.json` records the scenario, controller, total reward, step count, and success flag. World-model training expects numbered scenario directories such as `data/rollouts/scenario_0/` and `scenario_1/`.

## Development Environment

The recommended environment is the CUDA-enabled devcontainer, based on PyTorch 2.7 with CUDA 12.8. It requires a host NVIDIA driver and the NVIDIA container runtime.

Install the package in editable mode inside the container:

```bash
pip install -e .
rwm --help
pytest tests/unit
```

Useful commands include:

```bash
rwm collect run --help
rwm collect bulk --help
rwm inspect-dataset data/rollouts/scenario_0
rwm train-rwm --help
rwm train-controller --help
```

For human rendering from the container, grant local X11 access on the host:

```bash
xhost +local:root
```

The `nouveau` `libGL` warning sometimes emitted by CarRacing can be ignored when rendering otherwise works through the host NVIDIA setup.

## Project Status

The data collection and core model components are present, with useful unit and integration tests. However, the repository is an active research prototype: the Transformer migration is incomplete, reward gradients do not yet train the full perception path, evaluation is not fully deterministic, and behavior memory relies on overly strict exact latent-state matching.

The next phase is expected to redesign important structural components rather than merely finish the existing refactor. In particular, memory should represent similarity between situations instead of exact identity, and the complete model should eventually be trainable from reward back through the controller, temporal model, attention bottleneck, and encoder.
