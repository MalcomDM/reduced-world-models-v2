"""World-model trainer with corrected timing contract.

Timing (approved):
    belief b_t = Transformer(obs[t], action[t-1], history)
    ControllerTrunk.predict_reward(encode(b_t), action[t])  → reward[t]

Full-sequence training processes all timesteps in a single Transformer
pass after running perception once per frame.
"""

import time
import csv
from tqdm import tqdm
from pathlib import Path
from collections import deque
from typing import Tuple, List, Dict, Optional

import torch
from torch import optim, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F

from rwm.config.config import ACTION_DIM
from rwm.config.experiment_config import ExperimentConfig
from rwm.types import RolloutSample, WorldModelOutput
from rwm.models.rwm.model import ReducedWorldModel
from rwm.utils.checkpointing import save_checkpoint


def kl_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL( N(mu, sigma^2) || N(0, I) ) per posterior element.

    The KL expression is applied **elementwise** before any reduction:

        kl_per_element = 0.5 * (mu^2 + exp(logvar) - 1 - logvar)

    The result is then averaged over all dimensions (batch, time, patches,
    channels).  This is mathematically required: averaging mu or logvar
    before the nonlinear KL expression would give a different (incorrect)
    value.

    Parameters
    ----------
    mu:
        Post-sampling tokenizer mean (``(B, ..., D)``).
    logvar:
        Post-sampling tokenizer log-variance (``(B, ..., D)``).

    Returns
    -------
    Scalar tensor — mean KL over all posterior elements.
    """
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    return kl.mean()


class WorldModelTrainer:
    def __init__(
        self,
        train_loader: DataLoader[RolloutSample],
        val_loader: Optional[DataLoader[RolloutSample]] = None,
        out_dir: Path = Path("runs/rwm_train"),
        sequence_len: int = 16,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 3e-4,
        beta: float = 1.0,
        config: Optional[ExperimentConfig] = None,
        dataset_manifest_ref: Optional[str] = None,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.sequence_len = sequence_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta
        self.config = config
        self.dataset_manifest_ref = dataset_manifest_ref
        self._global_step = 0
        self._last_train_reward_mean = 0.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ReducedWorldModel(action_dim=ACTION_DIM).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = out_dir / "metrics.csv"
        self.best_val_metric = float("inf")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_one_epoch(self) -> Tuple[float, float, float, float]:
        self.model.train()
        total_losses: List[float] = []
        mse_losses: List[float] = []
        kl_losses: List[float] = []
        running_losses: deque[float] = deque(maxlen=20)
        reward_sum = 0.0
        reward_count = 0

        start = time.time()
        progress = tqdm(self.train_loader, desc="Training", leave=False)

        for step, batch in enumerate(progress):
            rewards = batch["reward"][:, :self.sequence_len]
            reward_sum += float(rewards.sum().item())
            reward_count += rewards.numel()

            loss_total, loss_mse, loss_kl = self._compute_batch_loss(batch)
            self.optimizer.zero_grad()
            loss_total.backward()
            self._log_grad_norms(step)
            self.optimizer.step()

            total_val = loss_total.item()
            total_losses.append(total_val)
            mse_losses.append(loss_mse.item())
            kl_losses.append(loss_kl.item() if isinstance(loss_kl, torch.Tensor) and loss_kl.numel() > 0 else 0.0)
            running_losses.append(total_val)

            if step % 5 == 0:
                avg = sum(running_losses) / len(running_losses)
                progress.set_postfix({"avg_loss": f"{avg:.4f}"})

        self._last_train_reward_mean = reward_sum / max(1, reward_count)
        n = len(total_losses)
        return (
            sum(total_losses) / max(1, n),
            sum(mse_losses) / max(1, n),
            sum(kl_losses) / max(1, n),
            time.time() - start,
        )

    def _log_grad_norms(self, step: int) -> None:
        """Compute and store per-block gradient norms after backward."""
        blocks = {
            "encoder": self.model.encoder,
            "tokenizer": self.model.tokenizer,
            "scorer": self.model.scorer,
            "selector": self.model.selector,
            "spatial_hd": self.model.spatial_hd,
            "world_hd": self.model.world_hd,
            "controller": self.model.controller,
        }
        norms = {}
        for name, module in blocks.items():
            total = 0.0
            count = 0
            for p in module.parameters():
                if p.grad is not None:
                    total += p.grad.norm().item() ** 2
                    count += 1
            norms[name] = (total ** 0.5) if count else 0.0
        # Store for logging at epoch end.
        if not hasattr(self, "_grad_norms"):
            self._grad_norms: Dict[str, List[float]] = {k: [] for k in blocks}
        for k, v in norms.items():
            self._grad_norms[k].append(v)

    def _compute_batch_loss(
        self, batch: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Full-sequence loss, returns (total_loss, reward_mse, kl_loss).

        prev_actions[:, 0] = predecessor_action
        prev_actions[:, t] = actions[:, t-1]
        Transformer(obs[:, t], prev_actions[:, t], history) → belief_t
        ControllerTrunk(belief_t, current_actions[:, t])    → reward_pred[t]
        MSE(reward_pred[t], rewards[:, t])
        """
        obs = batch["obs"].to(self.device, non_blocking=True)       # (B, T, C, H, W)
        act = batch["action"].to(self.device, non_blocking=True)    # (B, T, A)
        rew = batch["reward"].to(self.device, non_blocking=True)    # (B, T)

        B, T = rew.shape
        assert self.sequence_len <= T

        # Previous actions: first position uses the predecessor action
        # (zeros at true episode start, action[offset-1] mid-episode).
        if "predecessor_action" not in batch:
            raise KeyError(
                "Batch is missing required predecessor_action. RolloutDataset must "
                "supply action[offset - 1] (or zeros only at a true episode start)."
            )
        prev_actions = torch.zeros(B, self.sequence_len, ACTION_DIM, device=self.device)
        prev_actions[:, 0] = batch["predecessor_action"].to(
            self.device, non_blocking=True,
        )
        if self.sequence_len > 1:
            prev_actions[:, 1:] = act[:, :self.sequence_len - 1]
        current_actions = act[:, :self.sequence_len]

        out: WorldModelOutput = self.model.forward_sequence(
            obs[:, :self.sequence_len],
            prev_actions, current_actions,
            force_keep_input=True,
        )

        r_pred_seq = out.reward_pred_seq
        r_true_seq = rew[:, :self.sequence_len]
        loss_mse = F.mse_loss(r_pred_seq, r_true_seq)

        loss_kl = torch.zeros((), device=self.device)
        if out.tok_mu is not None and out.tok_logvar is not None:
            loss_kl = kl_normal(out.tok_mu, out.tok_logvar)

        return loss_mse + self.beta * loss_kl, loss_mse, loss_kl

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, float]:
        loader = self.val_loader if self.val_loader is not None else self.train_loader
        self.model.eval()
        total_mse, total_abs_error, total_baseline_sse, total_count = 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                obs = batch["obs"].to(self.device)
                act = batch["action"].to(self.device)
                rew = batch["reward"].to(self.device)

                B, T = rew.shape
                seq_len = min(self.sequence_len, T)

                if "predecessor_action" not in batch:
                    raise KeyError(
                        "Batch is missing required predecessor_action. RolloutDataset must "
                        "supply action[offset - 1] (or zeros only at a true episode start)."
                    )
                prev_actions = torch.zeros(B, seq_len, ACTION_DIM, device=self.device)
                prev_actions[:, 0] = batch["predecessor_action"].to(self.device)
                if seq_len > 1:
                    prev_actions[:, 1:] = act[:, :seq_len - 1]
                current_actions = act[:, :seq_len]

                out = self.model.forward_sequence(
                    obs[:, :seq_len], prev_actions, current_actions,
                    force_keep_input=True,
                )
                r_pred = out.reward_pred_seq
                r_true = rew[:, :seq_len]
                total_mse += F.mse_loss(r_pred, r_true, reduction="sum").item()
                total_abs_error += torch.abs(r_pred - r_true).sum().item()
                total_baseline_sse += (
                    (r_true - self._last_train_reward_mean).square().sum().item()
                )
                total_count += B * seq_len

        mse = total_mse / max(1, total_count)
        baseline_mse = total_baseline_sse / max(1, total_count)
        return {
            "val_mse": mse,
            "val_mae": total_abs_error / max(1, total_count),
            "mean_baseline_mse": baseline_mse,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def fit(self) -> Path:
        for epoch in range(1, self.epochs + 1):
            train_total, train_mse, train_kl, elapsed = self.train_one_epoch()
            val_metrics = self.evaluate()

            row: Dict[str, float] = {
                "epoch": epoch,
                "train_total": train_total,
                "train_mse": train_mse,
                "train_kl": train_kl,
                "val_mse": val_metrics["val_mse"],
                "val_mae": val_metrics["val_mae"],
                "baseline_mse": val_metrics["mean_baseline_mse"],
                "time": elapsed,
            }
            gn = getattr(self, "_grad_norms", None)
            if gn:
                for k, vals in gn.items():
                    if vals:
                        row[f"gn_{k}"] = vals[-1]
            self.log_and_checkpoint(epoch, row)

        return self.out_dir / "best_world_model.pt"

    def log_and_checkpoint(self, epoch: int, row: Dict[str, float]) -> None:
        write_header = not self.metrics_file.exists()
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self._global_step += 1
        val_metric = row["val_mse"]
        is_best = val_metric < self.best_val_metric

        if is_best:
            self.best_val_metric = val_metric
            torch.save(self.model.state_dict(), self.out_dir / "best_world_model.pt")

        if self.config is not None:
            ckpt_metrics = {"val_mse": val_metric, "train_mse": row["train_mse"]}
            if is_best:
                save_checkpoint(
                    path=self.out_dir / "checkpoint_best",
                    model_state=self.model.state_dict(),
                    optimizer_state=self.optimizer.state_dict(),
                    config=self.config,
                    global_step=self._global_step,
                    epoch=epoch,
                    metrics=ckpt_metrics,
                    dataset_manifest_ref=self.dataset_manifest_ref,
                )
            save_checkpoint(
                path=self.out_dir / "checkpoint_latest",
                model_state=self.model.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                config=self.config,
                global_step=self._global_step,
                epoch=epoch,
                metrics=ckpt_metrics,
                dataset_manifest_ref=self.dataset_manifest_ref,
            )

        if not is_best:
            print(f"[Epoch {epoch}] skipped ckpt: val_mse={val_metric:.4f} best={self.best_val_metric:.4f}")
        print(f"[Epoch {epoch}] train_mse={row['train_mse']:.4f} train_kl={row['train_kl']:.4f} val_mse={val_metric:.4f} time={row['time']:.1f}s")
