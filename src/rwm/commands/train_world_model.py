import typer
from typing import List
from pathlib import Path

from rwm.config.config import ERROR_THRESHOLD
from rwm.loops.train_world_model import train_world_model_loop


app = typer.Typer()


@app.command()
def train_world_model(
    base_dir: Path 		= typer.Argument(
									..., exists=True, file_okay=False, dir_okay=True,
									help="Base directory containing scenario_* subfolders"),
    max_scenario: int 	= typer.Argument(..., help="Train up to and including this scenario index (e.g. 2 means scenario_0 to scenario_2)" ),
    out_dir: Path 		= typer.Option( Path("runs/rwm_deterministic/loop_000"),
                                   file_okay=False, dir_okay=True, 
                                   help="Output directory for models and metrics" ),
    sequence_len: int 	= typer.Option(16, help="Number of timesteps per training sequence"),
    batch_size: int 	= typer.Option(32, help="Batch size for training"),
    max_epochs: int 	= typer.Option(50, help="Maximum number of training epochs"),
    image_size: int 	= typer.Option(64, help="Resize each frame to this square size"),
    alpha: float 		= typer.Option(1.0, help="Weight for cumulative reward loss component"),
    beta: float 		= typer.Option(0.1, help="Weight for per-step reward loss component"),
    error_threshold: float = typer.Option(ERROR_THRESHOLD, help="Early-stop when mean cumulative MAE < threshold"),
):
	""" Train a Reduced World Model using all scenario_i folders from scenario_0 to scenario_{max_scenario} under BASE_DIR. """
	out_dir.mkdir(parents=True, exist_ok=True)

	rollout_dirs: List[Path] = []
	for i in range(max_scenario + 1):
		scenario_dir = base_dir / f"scenario_{i}"
		if not scenario_dir.exists() or not scenario_dir.is_dir():
			typer.echo(f"❌ Missing scenario folder: {scenario_dir}", err=True)
			raise typer.Exit(1)
		rollout_dirs.append(scenario_dir)

	# Run the training loop
	best_model_path = train_world_model_loop(
		rollout_dirs=rollout_dirs,
		out_dir=out_dir,
		sequence_len=sequence_len,
		batch_size=batch_size,
		max_epochs=max_epochs,
		image_size=image_size,
		alpha=alpha,
		beta=beta,
		error_threshold=error_threshold,
	)

	typer.echo(f"✅ Training complete. Best model saved at: {best_model_path}")


if __name__ == "__main__":
    app()
