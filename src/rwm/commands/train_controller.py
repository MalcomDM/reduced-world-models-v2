import typer
from typer import Exit
from pathlib import Path
from typing import List

from rwm.loops.train_controller import ControllerTrainer
from rwm.models.controller.model import Controller
from rwm.models.rwm_deterministic.model import ReducedWorldModel

app = typer.Typer()


@app.command()
def train_controller(
    base_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Base directory containing scenario_* subfolders"
    ),
    max_scenario: int = typer.Argument(
        ..., help="Highest scenario index to include (e.g. 2 includes scenario_0,1,2)"
    ),
    model_ckpt: Path = typer.Option(
        ..., "--model-ckpt", exists=True, file_okay=True, dir_okay=False,
        help="Path to the trained world-model checkpoint (best_world_model.pt)",
    ),
    out_dir: Path = typer.Option(
        Path("runs/controller/loop_000"), file_okay=False, dir_okay=True,
        help="Where to save controller.pt and metrics.csv"
    ),
    warmup_steps: int = typer.Option(5, help="Number of real frames to warm up the latent state"),
    rollout_len: int = typer.Option(20, help="Number of imagined steps per rollout"),
    n_rollouts: int = typer.Option(100, help="Rollouts to simulate each epoch"),
    top_k: int = typer.Option(10, help="How many top rollouts to use for training"),
    epochs: int = typer.Option(20, help="Training epochs"),
    noise_std: float = typer.Option(0.1, help="Stddev of Gaussian noise for exploration"),
    lr: float = typer.Option(1e-3, help="Learning rate for Adam optimizer"),
    device: str = typer.Option("cuda", help="Torch device, e.g. 'cuda' or 'cpu'")
):
    """ Train the controller via imagined rollouts from a reduced world model. """
    scenario_dirs: List[str] = []
    for i in range(max_scenario + 1):
        dir_i = base_dir / f"scenario_{i}"
        if not dir_i.is_dir():
            typer.echo(f"❌ Missing scenario folder: {dir_i}", err=True)
            raise Exit(code=1)
        scenario_dirs.append(str(dir_i))

    # ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # instantiate trainer
    trainer = ControllerTrainer(
        model=ReducedWorldModel(),
        model_ckpt=str(model_ckpt),
        controller=Controller(),
        scenarios_dirs=scenario_dirs,
        out_dir=str(out_dir),
        warmup_steps=warmup_steps,
        rollout_len=rollout_len,
        noise_std=noise_std,
        lr=lr,
        device=device
    )

    # run training
    trainer.train(
        n_rollouts=n_rollouts,
        top_k=top_k,
        epochs=epochs
    )

if __name__ == "__main__":
    app()
