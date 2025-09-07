import time, csv
from tqdm import tqdm
from pathlib import Path
from collections import deque
from typing import Tuple, List, Dict

import torch
from torch import optim, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F

from rwm.config.config import ACTION_DIM, WRNN_HIDDEN_DIM
from rwm.types import RolloutSample
from rwm.models.rwm.model import ReducedWorldModel


class WorldModelTrainer:
	def __init__(
		self,
        loader: DataLoader[RolloutSample],
		out_dir: Path,
		sequence_len: int = 16,
		epochs: int = 10,
		batch_size: int = 32,
		lr: float = 3e-4,
		alpha: float = 1.0,
		beta: float = 1.0,
		 warmup_steps: int = 20,
	) -> None:
		self.loader  = loader
		self.sequence_len = sequence_len
		self.epochs, self.batch_size = epochs, batch_size
		self.lr, self.alpha, self.beta = lr, alpha, beta
		self.warmup_steps = warmup_steps

		# Model, optimizer, device
		self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model     = ReducedWorldModel(action_dim=ACTION_DIM).to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

		# Metrics state
		self.out_dir = out_dir
		out_dir.mkdir(exist_ok=True, parents=True)
		self.metrics_file = out_dir / "metrics.csv"
		self.best_loss = float("inf")


	def train_one_epoch(self) -> Tuple[float, float]:
		self.model.train()
		losses: List[float] = []
		running_losses: deque[float] = deque(maxlen=20)

		start = time.time()
		progress = tqdm(self.loader, desc="Training", leave=False)

		for step, batch in enumerate(progress):
			loss = self._compute_batch_loss(batch)
			self.optimizer.zero_grad()
			loss.backward()						# type: ignore
			self.optimizer.step()				# type: ignore

			loss_val = loss.item()
			losses.append(loss_val)
			running_losses.append(loss_val)

			if step % 5 == 0:
				running_loss_avg: float = sum(running_losses)/len(running_losses)
				progress.set_postfix({ "avg_loss": f"{running_loss_avg:.4f}" })	# type: ignore

		mean_loss: float = float(sum(losses) / len(losses))
		return mean_loss, time.time() - start


	def _compute_batch_loss(self, batch: Dict[str, Tensor]) -> Tensor:
		"""Compute average per-timestep MSE loss for a batch."""
		B = batch["reward"].shape[0]
		h = torch.zeros(B, WRNN_HIDDEN_DIM, device=self.device)
		c = torch.zeros(B, WRNN_HIDDEN_DIM, device=self.device)

		obs = batch["obs"].to(self.device, non_blocking=True)		# (B, T, C, H, W)
		act = batch["action"].to(self.device, non_blocking=True)	# (B, T, 3)
		rew = batch["reward"].to(self.device, non_blocking=True)	# (B, T)

		loss_seq: Tensor = torch.tensor(0.0, device=self.device)
		a_prev = torch.zeros(B, ACTION_DIM, device=self.device)
		for t in range(self.sequence_len):
			img_t = obs[:, t]							# (B, C, H, W)
			r_true_t = rew[:, t].unsqueeze(-1)			# (B, 1) for proper shape
			force_keep = (t < self.warmup_steps)
			h, c, r_pred_t, *_ = self.model(img=img_t, a_prev=a_prev, h_prev=h, c_prev=c, force_keep_input=force_keep)
			loss_seq += F.mse_loss(r_pred_t, r_true_t)
			a_prev = act[:, t]

		return loss_seq / self.sequence_len


	def evaluate(self) -> Dict[str, float]:
		"""
		Run one pass in eval mode, returning:
			- mae_cum: mean absolute error on total reward per rollout
			- mae_step: mean absolute error per timestep
		"""
		self.model.eval()
		total_cum, total_step, total_count = 0.0, 0.0, 0

		with torch.no_grad():
			for batch in tqdm( self.loader, desc="Evaluating", leave=False ):
				cum, step, count = self._compute_eval_batch(batch)
				total_cum += cum
				total_step += step
				total_count += count

		return {
			"mae_cum": total_cum / total_count,
			"mae_step": total_step / (total_count * self.sequence_len),
		}
	

	def _compute_eval_batch(self, batch: RolloutSample) -> Tuple[float, float, int]:
		"""Compute (cum_error, step_error, batch_size) for one eval batch."""
		B = batch["reward"].shape[0]
		h = torch.zeros(B, WRNN_HIDDEN_DIM, device=self.device)
		c = torch.zeros(B, WRNN_HIDDEN_DIM, device=self.device)

		obs = batch["obs"].to(self.device)
		act = batch["action"].to(self.device)
		rew = batch["reward"].to(self.device)

		preds: List[Tensor] = []
		for t in range(self.sequence_len):
			h, c, r_pred, *_ = self.model(obs[:, t], act[:, t], h, c)
			preds.append(r_pred.squeeze(-1))

		r_pred_seq = torch.stack(preds, dim=1)
		cum_error  = torch.abs(rew.sum(1)    - r_pred_seq.sum(1)).sum().item()
		step_error = torch.abs(rew - r_pred_seq).sum().item()
		return cum_error, step_error, B
	

	def fit(self) -> Path:
		"""
		Fixed-epoch fit:
			- calls train_one_epoch()
			- calls evaluate()
			- logs via _log_and_checkpoint()
			- saves best by train_loss
		"""
		for epoch in range(1, self.epochs + 1):
			train_loss, elapsed = self.train_one_epoch()
			eval_metrics = self.evaluate()

			row: Dict[str, float] = {
				"epoch": epoch,
				"train_loss": train_loss,
				"mae_cum": eval_metrics["mae_cum"],
				"mae_step": eval_metrics["mae_step"],
				"time": elapsed,
			}
			self.log_and_checkpoint(epoch, row)

		return self.out_dir / "best_world_model.pt"


	def log_and_checkpoint(self, epoch: int, row: Dict[str, float]) -> None:
		"""Append metrics row and save best model by train_loss."""
		write_header = not self.metrics_file.exists()
		with open(self.metrics_file, "a", newline="") as f:
			writer = csv.DictWriter(f, fieldnames=list(row.keys()))
			if write_header:
				writer.writeheader()
			writer.writerow(row)

		loss = row["train_loss"]
		if loss < self.best_loss:
			self.best_loss = loss
			torch.save(self.model.state_dict(), self.out_dir / "best_world_model.pt")			# type: ignore
		else:
			print(f"[Epoch {epoch}] skipped ckpt: loss={loss:.4f} best={self.best_loss:.4f}")

		print(f"[Epoch {epoch}] train_loss={loss:.4f} "
				f"mae_cum={row['mae_cum']:.2f} time={row['time']:.1f}s")