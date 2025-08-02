
import typer

from rwm.commands import collect, inspect_dataset


app = typer.Typer()
app.add_typer(collect.app, 			name="collect")
app.add_typer(inspect_dataset.app,	name="inspect-dataset")
