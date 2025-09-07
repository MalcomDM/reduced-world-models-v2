import typer
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rwm.config.config import WRNN_HIDDEN_DIM
from rwm.data.rollout_dataset import RolloutDataset
from rwm.models.rwm.model import ReducedWorldModel


app = typer.Typer()


@app.command()
def run_eval(
    model_path: Path 	= typer.Argument(..., help="Path to the trained .pt model"),
    base_dir: Path 		= typer.Argument(..., exists=True, file_okay=False, dir_okay=True,
									help="Base directory containing scenario_* subfolders"),
    scenario: int 		= typer.Argument(..., help="Train up to and including this scenario index (e.g. 2 means scenario_0 to scenario_2)" ),
    output_csv: Path 	= typer.Option("reward_eval.csv", help="Where to save true vs predicted rewards"),
    sequence_len: int 	= typer.Option(16),
    image_size: int 	= typer.Option(64),
    batch_size: int 	= typer.Option(8),
):
	"""Evaluate a trained RWM on a rollout directory and log predicted vs true rewards."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = ReducedWorldModel(action_dim=3).to(device)
	ckpt = torch.load(model_path, map_location=device)
	state = ckpt.get("model_state", ckpt)
	model.load_state_dict(state)
	model.eval()

	scenario_dir = base_dir / f"scenario_{scenario}"
	if not scenario_dir.exists() or not scenario_dir.is_dir():
		typer.echo(f"❌ Missing scenario folder: {scenario_dir}", err=True)
		raise typer.Exit(1)

	dataset = RolloutDataset(scenario_dir, sequence_len=sequence_len, image_size=image_size)
	loader = DataLoader(dataset, batch_size=batch_size)

	with open(output_csv, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["rollout_idx", "step", "r_true", "r_pred"])

		with torch.no_grad():
			for i, batch in enumerate(loader):
				B = batch["reward"].shape[0]
				h = torch.zeros(B, WRNN_HIDDEN_DIM, device=device)
				c = torch.zeros_like(h)
				obs = batch["obs"].to(device)
				act = batch["action"].to(device)
				rew = batch["reward"].to(device)

				for t in range(sequence_len):
					h, c, r_pred, *_ = model(obs[:, t], act[:, t], h, c)
					for b in range(B):
						writer.writerow([i * batch_size + b, t, rew[b, t].item(), r_pred[b].item()])
				if i > 1:
					break

	typer.echo(f"✅ Evaluation saved to {output_csv}")
