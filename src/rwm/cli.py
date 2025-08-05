
import typer

from rwm.commands import collect, inspect_dataset
from rwm.commands.train_world_model import train_world_model
from rwm.commands.train_controller import train_controller
from rwm.commands.rwm_manual_test import run
from rwm.commands.evaluate_rwm_on_rollouts import run_eval


app = typer.Typer()
app.add_typer(collect.app, 				name="collect")
app.add_typer(inspect_dataset.app,		name="inspect-dataset")

app.command(name="train-rwm")(train_world_model)
app.command(name="train-controller")(train_controller)

app.command(name="test-rwm-manually")(run)
app.command(name="test-rwm-rollouts")(run_eval)