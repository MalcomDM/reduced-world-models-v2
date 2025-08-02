import typer, torch
import numpy as np
from torch import Tensor
from pathlib import Path
from rich import print
from rich.table import Table

from rwm.datasets.rollout_dataset import RolloutDataset


app = typer.Typer()


@app.command()
def inspect(
    root_dir: 		Path = typer.Argument(..., help="Path to rollout data"),
    sequence_len: 	 int = typer.Option(16, help="Length of each sampled sequence"),
    image_size: 	 int = typer.Option(64, help="Size to resize each observation"),
    show_samples: 	 int = typer.Option(5, help="How many samples to preview"),
    include_done: 	bool = typer.Option(False, help="Include sequences that span done=True"),
):
	"""Inspect rollout dataset statistics and example rewards."""
	ds = RolloutDataset(
		root_dir=root_dir,
		sequence_len=sequence_len,
		image_size=image_size,
		include_done=include_done,
	)

	print(f"\n[bold cyan]✅ RolloutDataset loaded[/bold cyan]")
	print(f"Found [bold]{len(ds)}[/bold] sequences in {root_dir}")
	print(f"Sequence length: {sequence_len}, Image size: {image_size}x{image_size}")

	rewards: list[Tensor] = []
	dones: list[Tensor] = []

	for path, offset in ds.samples:
		with np.load(path) as data:
			reward_slice: Tensor = torch.tensor(
				data["reward"][offset : offset + ds.sequence_len],
				dtype=torch.float32
			)
			done_slice: Tensor = torch.tensor(
				data["done"][offset : offset + ds.sequence_len],
				dtype=torch.bool
			)
			rewards.append(reward_slice)
			dones.append(done_slice)

	reward_tensor: Tensor = torch.stack(rewards)  # shape (N, T)
	done_tensor: Tensor = torch.cat(dones)        # shape (N * T,)

	reward_sum: Tensor = reward_tensor.sum(dim=1)
	reward_mean: float = reward_sum.mean().item()
	reward_std: float = reward_sum.std().item()
	done_rate: float = done_tensor.float().mean().item()

	print(f"\n[bold]Total reward stats per sequence:[/bold]")
	print(f"- Mean: {reward_mean:.2f}")
	print(f"- Std:  {reward_std:.2f}")
	print(f"- Average done ratio per step: {done_rate:.2%}")

	print(f"\n[bold]Sample reward sequences:[/bold]")
	table = Table(show_header=True, header_style="bold magenta")
	table.add_column("Idx", justify="right")
	table.add_column("Reward sequence")
	table.add_column("Done flags")

	for i in range(min(show_samples, len(ds))):
		reward_tensor = ds[i]["reward"]
		done_tensor   = ds[i]["done"]

		r: list[float] = [float(v) for v in reward_tensor]
		d: list[bool]  = [bool(v)  for v in done_tensor]

		table.add_row(
			str(i),
			f"{[round(v, 2) for v in r]}",
			f"{d}"
		)
	print(table)