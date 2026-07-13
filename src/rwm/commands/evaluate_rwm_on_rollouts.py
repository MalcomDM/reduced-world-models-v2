import typer
from pathlib import Path


app = typer.Typer()


@app.command()
def run_eval(
    model_path: Path = typer.Argument(
        ..., exists=True, help="Path to the trained .pt model"
    ),
    base_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Base directory containing scenario_* subfolders",
    ),
    scenario: int = typer.Argument(
        ...,
        help=(
            "Train up to and including this scenario index "
            "(e.g. 2 means scenario_0 to scenario_2)"
        ),
    ),
    output_csv: Path = typer.Option(
        "reward_eval.csv",
        help="Where to save true vs predicted rewards",
    ),
    sequence_len: int = typer.Option(16),
    image_size: int = typer.Option(64),
    batch_size: int = typer.Option(8),
):
    """Evaluate a trained RWM on a rollout directory.

    .. deprecated::
       This command was written for the legacy LSTM model and calls an
       incompatible interface (``h_prev, c_prev``) that does not match
       the current ``ReducedWorldModel`` which uses a
       ``CausalTransformer``.

       The command will be rewritten in Stage 1/2 when the evaluation
       loop is updated to use the Transformer's history interface.
    """
    typer.echo(
        "ERROR: test-rwm-rollouts uses a legacy LSTM interface "
        "(h_prev, c_prev) that is incompatible with the current "
        "CausalTransformer-based model.\n\n"
        "This command will be restored in a later stage when the "
        "evaluation loop is migrated to use the Transformer's history "
        "buffer interface.\n\n"
        "To evaluate the model, use the training loop's built-in "
        "evaluation or write a custom eval script using "
        "ReducedWorldModel with WorldModelOutput.",
        err=True,
    )
    raise typer.Exit(1)
