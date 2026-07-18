"""CLI for world-model training with structured run artifacts.

Output policy (Stage 2 approved):
- Default: ``runs/<experiment>/<timestamp>/`` structured directory.
- Explicit ``--out-dir``: use exactly that directory.
- Reject simultaneous ``--out-dir`` and ``--run-id`` as ambiguous.
"""

from typing import List, Optional
from pathlib import Path

import typer

from rwm.config.experiment_config import (
    ExperimentConfig,
    DataConfig,
    TemporalConfig,
    PerceptionConfig,
    ControllerConfig,
    TrainingConfig,
    TemporalMaskConfig,
)
from rwm.loops.train_world_model import train_world_model_loop
from rwm.utils.run_directory import create_run_directory, _RUNS_ROOT
from rwm.utils.seeding import set_seed


app = typer.Typer()


@app.command()
def train_world_model(
    base_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Base directory containing scenario_* subfolders",
    ),
    max_scenario: int = typer.Argument(
        ...,
        help="Train up to scenario index N (e.g. 2 means scenario_0 to scenario_2)",
    ),
    out_dir: Optional[Path] = typer.Option(
        None, file_okay=False, dir_okay=True,
        help="Exact output directory (overrides default run dir). "
             "Cannot be used with --run-id.",
    ),
    sequence_len: int = typer.Option(
        16, help="Number of timesteps per training sequence",
    ),
    batch_size: int = typer.Option(
        32, help="Batch size for training",
    ),
    max_epochs: int = typer.Option(
        50, help="Maximum number of training epochs",
    ),
    image_size: int = typer.Option(
        64, help="Resize each frame to this square size",
    ),
    beta: float = typer.Option(
        1.0, help="Weight for KL divergence loss",
    ),
    warmup_steps: int = typer.Option(
        5, help="Warmup steps without observational dropout",
    ),
    val_ratio: float = typer.Option(
        0.2, help="Fraction of rollout files held out for validation",
    ),
    seed: Optional[int] = typer.Option(
        42,
        help="Random seed for reproducibility. "
             "Default 42 preserves current behavior.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        help="Explicit run identifier. Cannot be used with --out-dir.",
    ),
    experiment_name: str = typer.Option(
        "train-rwm",
        help="Experiment name (used in default run directory path).",
    ),
    cache_dir: Optional[Path] = typer.Option(
        None, file_okay=False, dir_okay=True,
        help="Path to frame cache (default: no cache). "
             "Build with: python scripts/build_frame_cache.py",
    ),
    reward_head_kind: str = typer.Option(
        "linear", help="Reward head architecture: 'linear' or 'nonlinear'",
    ),
    reward_head_hidden_dim: int = typer.Option(
        32, help="Hidden dimension for nonlinear reward head",
    ),
    selection_mode: str = typer.Option(
        "learned", help="Patch selection: learned | fixed_uniform | fixed_random",
    ),
    selection_k: int = typer.Option(
        8, help="Number of selected patches",
    ),
    selection_seed: int = typer.Option(
        0, help="RNG seed for fixed_random selection",
    ),
    tokenizer_eval_mode: str = typer.Option(
        "sample", help="Tokenizer evaluation policy: sample or mean",
    ),
    temporal_mask_enabled: bool = typer.Option(
        False, help="Enable temporal observation masking during training (D.1)",
    ),
    temporal_mask_warmup: int = typer.Option(
        4, help="Visible warmup steps before mask starts",
    ),
    temporal_mask_horizons: str = typer.Option(
        "1,2,4,8,12", help="Comma-separated approved mask horizons",
    ),
    temporal_mask_probability: float = typer.Option(
        0.5, help="Target per-sample mask probability after ramp",
    ),
    temporal_mask_ramp_epochs: int = typer.Option(
        2, help="Epochs over which mask probability ramps from 0 to target",
    ),
):
    """Train a ReducedWorldModel on rollout data with held-out validation.

    By default creates a structured run directory under
    ``runs/<experiment>/<timestamp>/`` with persisted config,
    environment metadata, dataset manifest, and structured checkpoints.

    Use ``--out-dir`` to specify an exact output directory instead.
    ``--out-dir`` and ``--run-id`` cannot be used together.
    """
    # --- Validate output options ---
    if out_dir is not None and run_id is not None:
        typer.echo(
            "ERROR: --out-dir and --run-id are mutually exclusive. "
            "Use --out-dir for an exact output path, or --run-id for "
            "a structured run under the default directory.",
            err=True,
        )
        raise typer.Exit(1)

    # --- Build resolved config ---
    cfg = ExperimentConfig(
        experiment_name=experiment_name,
        run_id=run_id or "",
        seed=seed if seed is not None else 42,
        data=DataConfig(
            dataset_dir=str(base_dir.resolve()),
            sequence_len=sequence_len,
            image_size=image_size,
            cache_dir=str(cache_dir.resolve()) if cache_dir is not None else "",
        ),
        temporal=TemporalConfig(
            seq_len=sequence_len,
            warmup_steps=warmup_steps,
        ),
        perception=PerceptionConfig(
            k=selection_k,
            selection_mode=selection_mode,
            selection_seed=selection_seed,
            tokenizer_eval_mode=tokenizer_eval_mode,
        ),
        controller=ControllerConfig(
            reward_head_kind=reward_head_kind,
            reward_head_hidden_dim=reward_head_hidden_dim,
        ),
        training=TrainingConfig(
            batch_size=batch_size,
            max_epochs=max_epochs,
            beta=beta,
            temporal_mask=TemporalMaskConfig(
                enabled=temporal_mask_enabled,
                warmup_steps=temporal_mask_warmup,
                horizons=[int(x) for x in temporal_mask_horizons.split(",")],
                target_mask_probability=temporal_mask_probability,
                ramp_epochs=temporal_mask_ramp_epochs,
            ),
        ),
    )

    set_seed(cfg.seed)

    # --- Resolve output directory ---
    if out_dir is not None:
        # Explicit --out-dir: use exactly this path.
        resolved_out = out_dir.resolve()
        resolved_out.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Output directory: {resolved_out}")
    else:
        # Default structured run directory.
        resolved_out = create_run_directory(
            experiment_name=experiment_name,
            config=cfg,
            run_id=run_id,
        )
        typer.echo(f"Run directory: {resolved_out}")

    # --- Collect rollout directories ---
    rollout_dirs: List[Path] = []
    for i in range(max_scenario + 1):
        scenario_dir = base_dir / f"scenario_{i}"
        if not scenario_dir.exists() or not scenario_dir.is_dir():
            typer.echo(f"Missing scenario folder: {scenario_dir}", err=True)
            raise typer.Exit(1)
        rollout_dirs.append(scenario_dir)

    # --- Train ---
    best_model_path = train_world_model_loop(
        rollout_dirs=rollout_dirs,
        out_dir=resolved_out,
        sequence_len=sequence_len,
        batch_size=batch_size,
        max_epochs=max_epochs,
        image_size=image_size,
        beta=beta,
        warmup_steps=warmup_steps,
        val_ratio=val_ratio,
        config=cfg,
        cache_dir=cache_dir,
    )

    typer.echo(f"Training complete. Best model: {best_model_path}")
