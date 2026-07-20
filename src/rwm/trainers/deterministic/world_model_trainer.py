"""World-model trainer with corrected timing contract.

Timing (approved):
    belief b_t = Transformer(obs[t], action[t-1], history)
    ControllerTrunk.predict_reward(encode(b_t), action[t])  → reward[t]

SRU burn-in mode:
    The full sequence includes a burn-in prefix.  valid_step and loss_mask
    indicate which positions are padding vs normal, and which contribute to
    the loss.  Target loss gradients flow through the burn-in state
    construction — no detach or no_grad at the burn-in/target boundary.
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
from rwm.config.experiment_config import ExperimentConfig, TemporalMaskConfig, TemporalConfig
from rwm.types import RolloutSample, WorldModelOutput
from rwm.models.rwm.model import ReducedWorldModel
from rwm.utils.checkpointing import save_checkpoint
from rwm.trainers.deterministic.temporal_mask import (
    sample_mask,
    current_mask_probability,
    _validate_config as validate_mask_config,
)


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


def masked_kl_normal(mu: Tensor, logvar: Tensor, loss_mask: Tensor) -> Tensor:
    """Mean posterior KL over target positions only.

    Boolean indexing retains the patch and latent dimensions, giving each
    posterior element at every selected batch/time position equal weight.
    Returns scalar zero for an empty mask.
    """
    if loss_mask.dtype != torch.bool or loss_mask.shape != mu.shape[:2]:
        raise ValueError(
            "loss_mask must be bool with shape matching mu batch/time dimensions"
        )
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    selected = kl[loss_mask]
    if selected.numel() == 0:
        return torch.zeros((), device=mu.device)
    return selected.mean()


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
        self._train_rng = torch.Generator(device="cpu")
        self._train_rng.manual_seed(0)  # deterministic reproducibility

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Read temporal-mask config.
        self._mask_cfg: TemporalMaskConfig = TemporalMaskConfig()
        if config is not None and hasattr(config, "training"):
            tc = config.training
            if hasattr(tc, "temporal_mask"):
                self._mask_cfg = tc.temporal_mask
        if self._mask_cfg.enabled:
            validate_mask_config(
                self._mask_cfg.warmup_steps,
                self._mask_cfg.horizons,
                self.sequence_len,
            )

        # Read temporal config for backend selection.
        self._temporal_cfg: TemporalConfig = TemporalConfig()
        if config is not None and hasattr(config, "temporal"):
            self._temporal_cfg = config.temporal
        self._is_sru = self._temporal_cfg.backend == "minimal_sru"

        # Read architecture config.
        rh_kind = "linear"
        rh_hidden = 32
        sel_mode = "learned"
        sel_k = 8
        sel_seed = 0
        tok_eval_mode = "sample"
        if config is not None:
            if hasattr(config, "controller"):
                cc = config.controller
                rh_kind = getattr(cc, "reward_head_kind", "linear")
                rh_hidden = getattr(cc, "reward_head_hidden_dim", 32)
            if hasattr(config, "perception"):
                pc = config.perception
                sel_mode = getattr(pc, "selection_mode", "learned")
                sel_k = getattr(pc, "k", 8)
                sel_seed = getattr(pc, "selection_seed", 0)
                tok_eval_mode = getattr(pc, "tokenizer_eval_mode", "sample")
        self.model = ReducedWorldModel(
            action_dim=ACTION_DIM,
            reward_head_kind=rh_kind,
            reward_head_hidden_dim=rh_hidden,
            selection_mode=sel_mode,
            selection_k=sel_k,
            selection_seed=sel_seed,
            tokenizer_eval_mode=tok_eval_mode,
            temporal_config=self._temporal_cfg,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self._current_mask_prob: float = 0.0

        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = out_dir / "metrics.csv"
        self.best_val_metric = float("inf")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_epoch_sequential_tbptt(self, epoch: int = 1) -> Tuple[float, float, float, float]:
        """One epoch of sequential truncated-BPTT training.

        Iterates over the multi-stream sequential dataset.  Carries SRU
        state across consecutive chunks of the same episode.  Detaches
        carried state after each optimizer step.  Resets to zeros when
        ``episode_start`` is True.
        """
        self.model.train()
        total_losses: List[float] = []
        mse_losses: List[float] = []
        kl_losses: List[float] = []
        running_losses: deque[float] = deque(maxlen=20)
        reward_sum = 0.0
        reward_count = 0
        processed_real_transitions = 0
        processed_model_positions = 0

        # Carried SRU states per batch slot; reset is handled via episode_start.
        carried_z: Optional[Tensor] = None

        def _to_dev(t: Tensor) -> Tensor:
            return t.to(self.device, non_blocking=True) if t.device != self.device else t

        start = time.time()
        progress = tqdm(self.train_loader, desc="Training", leave=False)

        for step, batch in enumerate(progress):
            obs = batch["obs"].to(self.device, non_blocking=True)
            act = batch["action"].to(self.device, non_blocking=True)
            rew = batch["reward"].to(self.device, non_blocking=True)
            vs_t = batch.get("valid_step")
            valid_step = _to_dev(vs_t) if vs_t is not None else None
            lm_t = batch.get("loss_mask")
            loss_mask = _to_dev(lm_t) if lm_t is not None else None
            es_t = batch.get("episode_start")
            episode_start = _to_dev(es_t) if es_t is not None else None

            B, T_full = rew.shape
            # Accumulate accounting.
            if valid_step is not None:
                real_this = valid_step.sum().item()
            else:
                real_this = B * T_full
            processed_real_transitions += int(real_this)
            processed_model_positions += B * T_full
            reward_sum += rew.sum().item() if loss_mask is None else \
                (rew * loss_mask.float()).sum().item()
            reward_count += real_this

            # Build prev_actions.
            if "predecessor_action" not in batch:
                raise KeyError("Missing predecessor_action")
            pred = batch["predecessor_action"].to(self.device, non_blocking=True)
            prev_actions = torch.zeros(B, T_full, ACTION_DIM, device=self.device)
            if T_full > 1:
                prev_actions[:, 1:] = act[:, :T_full - 1]
            # Override first valid position with predecessor.
            if valid_step is not None:
                fv = valid_step.long().argmax(dim=1)
                for b in range(B):
                    fv_b = fv[b].item()
                    if valid_step[b, fv_b]:
                        prev_actions[b, fv_b] = pred[b]
            else:
                prev_actions[:, 0] = pred
            current_actions = act

            # Reset states for streams starting a new episode.
            if episode_start is not None and carried_z is not None:
                cz = _to_dev(carried_z)
                for b in range(B):
                    if episode_start[b]:
                        cz[b] = 0.0
                carried_z = cz

            # Forward.
            out = self.model.forward_sequence(
                obs, prev_actions, current_actions,
                force_keep_input=True,
                temporal_state=carried_z,
                valid_step=valid_step,
            )
            carried_z = out.temporal_state.detach()  # detach after forward

            # Target-only MSE and KL.
            loss_mask_f = None
            if loss_mask is not None:
                loss_mask_f = loss_mask.float()
                diff = (out.reward_pred_seq - rew).pow(2)
                loss_mse = (diff * loss_mask_f).sum() / loss_mask_f.sum().clamp_min(1)
                loss_kl = masked_kl_normal(out.tok_mu, out.tok_logvar, loss_mask) \
                    if out.tok_mu is not None and out.tok_logvar is not None else \
                    torch.zeros((), device=self.device)
            else:
                loss_mse = F.mse_loss(out.reward_pred_seq, rew)
                loss_kl = torch.zeros((), device=self.device)
                if out.tok_mu is not None and out.tok_logvar is not None:
                    loss_kl = 0.5 * (out.tok_mu.pow(2) + out.tok_logvar.exp() - 1.0 - out.tok_logvar).mean()

            loss_total = loss_mse + self.beta * loss_kl

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
        self._seq_metrics = {
            "processed_real_transitions": processed_real_transitions,
            "processed_model_positions": processed_model_positions,
            "tbptt_steps": self._temporal_cfg.tbptt_steps,
            "sru_training_mode": self._temporal_cfg.sru_training_mode,
        }
        return (
            sum(total_losses) / max(1, n),
            sum(mse_losses) / max(1, n),
            sum(kl_losses) / max(1, n),
            time.time() - start,
        )

    def _train_epoch_macroblock_tbptt(self, epoch: int = 1) -> Tuple[float, float, float, float]:
        """One epoch of random-macroblock TBPTT training.

        Per macrobatch (B samples x (burn_in + target_total) positions):
          1. Forward burn-in once (keeps graph for first target chunk).
          2. For each target TBPTT chunk:
             a. forward chunk;
             b. loss on chunk target positions;
             c. backward and optimizer step;
             d. detach carried ``z``.

        The first target chunk's backward propagates through the burn-in
        graph.  Subsequent chunks receive a detached state.
        """
        self.model.train()
        total_losses: List[float] = []
        mse_losses: List[float] = []
        kl_losses: List[float] = []
        running_losses: deque[float] = deque(maxlen=20)
        reward_sum = 0.0
        reward_count = 0
        opt_updates = 0
        real_target_transitions = 0
        real_burn_in_transitions = 0
        processed_model_positions = 0
        direct_supervised_targets = 0

        burn_in = self._temporal_cfg.sru_burn_in_steps
        target_total = self._temporal_cfg.macroblock_target_steps
        chunk_len = self._temporal_cfg.tbptt_steps
        n_chunks = target_total // chunk_len

        start = time.time()
        progress = tqdm(self.train_loader, desc="Training", leave=False)

        for step, batch in enumerate(progress):
            obs = batch["obs"].to(self.device, non_blocking=True)
            act = batch["action"].to(self.device, non_blocking=True)
            rew = batch["reward"].to(self.device, non_blocking=True)
            vs_t = batch.get("valid_step")
            valid_step = vs_t.to(self.device, non_blocking=True) if vs_t is not None else None
            lm_t = batch.get("loss_mask")
            loss_mask = lm_t.to(self.device, non_blocking=True) if lm_t is not None else None
            bi_t = batch.get("burn_in_mask")
            burn_in_mask = bi_t.to(self.device, non_blocking=True) if bi_t is not None else None

            if "predecessor_action" not in batch:
                raise KeyError("Missing predecessor_action")
            pred = batch["predecessor_action"].to(self.device, non_blocking=True)

            B = obs.shape[0]
            total_T = obs.shape[1]

            # Build full-sequence prev_actions once.
            prev_actions = torch.zeros(B, total_T, ACTION_DIM, device=self.device)
            if total_T > 1:
                prev_actions[:, 1:] = act[:, :total_T - 1]
            if valid_step is not None:
                first_valid = valid_step.long().argmax(dim=1)
                for b in range(B):
                    fv = first_valid[b].item()
                    if valid_step[b, fv]:
                        prev_actions[b, fv] = pred[b]
            else:
                prev_actions[:, 0] = pred

            # ---- Accounting (from masks) ----
            if valid_step is not None and burn_in_mask is not None and loss_mask is not None:
                real_burn_in_this = int((burn_in_mask & valid_step).sum().item())
                real_target_this = int((loss_mask & valid_step).sum().item())
            else:
                real_burn_in_this = burn_in if burn_in > 0 else 0
                real_target_this = target_total
            real_burn_in_transitions += real_burn_in_this
            real_target_transitions += real_target_this
            if loss_mask is not None:
                direct_supervised_targets += int(loss_mask.sum().item())

            # ---- Phase 1: Burn-in forward (once, keep graph) ----
            z = None
            if burn_in > 0:
                out_bi = self.model.forward_sequence(
                    obs[:, :burn_in], prev_actions[:, :burn_in], act[:, :burn_in],
                    force_keep_input=True,
                    temporal_state=None,
                    valid_step=valid_step[:, :burn_in] if valid_step is not None else None,
                )
                z = out_bi.temporal_state
                processed_model_positions += B * burn_in

            # ---- Phase 2: Target TBPTT chunks ----
            for ci in range(n_chunks):
                cs = burn_in + ci * chunk_len
                ce = min(cs + chunk_len, total_T)
                if ce <= cs:
                    break

                chunk_obs = obs[:, cs:ce]
                chunk_prev = prev_actions[:, cs:ce]
                chunk_act = act[:, cs:ce]
                chunk_vs = valid_step[:, cs:ce] if valid_step is not None else None
                chunk_lm = loss_mask[:, cs:ce] if loss_mask is not None else None

                out_chunk = self.model.forward_sequence(
                    chunk_obs, chunk_prev, chunk_act,
                    force_keep_input=True,
                    temporal_state=z,
                    valid_step=chunk_vs,
                )
                z = out_chunk.temporal_state
                processed_model_positions += B * chunk_obs.shape[1]

                # Skip empty chunk (no valid target positions).
                if chunk_lm is not None and chunk_lm.sum() == 0:
                    z = z.detach()
                    continue

                chunk_rew = rew[:, cs:ce]

                if chunk_lm is not None:
                    lm_f = chunk_lm.float()
                    n_target = lm_f.sum().clamp_min(1)
                    diff = (out_chunk.reward_pred_seq - chunk_rew).pow(2)
                    loss_mse = (diff * lm_f).sum() / n_target
                    loss_kl = masked_kl_normal(out_chunk.tok_mu, out_chunk.tok_logvar, chunk_lm) \
                        if out_chunk.tok_mu is not None and out_chunk.tok_logvar is not None \
                        else torch.zeros((), device=self.device)
                else:
                    loss_mse = F.mse_loss(out_chunk.reward_pred_seq, chunk_rew)
                    loss_kl = torch.zeros((), device=self.device)
                    if out_chunk.tok_mu is not None and out_chunk.tok_logvar is not None:
                        loss_kl = 0.5 * (out_chunk.tok_mu.pow(2) + out_chunk.tok_logvar.exp() - 1.0 - out_chunk.tok_logvar).mean()

                loss_total = loss_mse + self.beta * loss_kl

                self.optimizer.zero_grad()
                loss_total.backward()
                self._log_grad_norms(step * n_chunks + ci)
                self.optimizer.step()

                reward_sum += (chunk_rew * (chunk_lm.float() if chunk_lm is not None else 1.0)).sum().item()
                reward_count += chunk_lm.sum().item() if chunk_lm is not None else chunk_rew.numel()
                opt_updates += 1
                total_losses.append(loss_total.item())
                mse_losses.append(loss_mse.item())
                kl_losses.append(loss_kl.item() if isinstance(loss_kl, torch.Tensor) and loss_kl.numel() > 0 else 0.0)
                running_losses.append(loss_total.item())

                z = z.detach()

                if (step * n_chunks + ci) % 5 == 0:
                    avg = sum(running_losses) / len(running_losses)
                    progress.set_postfix({"avg_loss": f"{avg:.4f}"})

        self._last_train_reward_mean = reward_sum / max(1, reward_count)
        n = len(total_losses)
        self._seq_metrics = {
            "opt_updates": opt_updates,
            "real_target_transitions": real_target_transitions,
            "real_burn_in_transitions": real_burn_in_transitions,
            "processed_model_positions": processed_model_positions,
            "direct_supervised_targets": direct_supervised_targets,
            "tbptt_steps": self._temporal_cfg.tbptt_steps,
            "sru_training_mode": self._temporal_cfg.sru_training_mode,
        }
        return (
            sum(total_losses) / max(1, n),
            sum(mse_losses) / max(1, n),
            sum(kl_losses) / max(1, n),
            time.time() - start,
        )

    def train_one_epoch(self, epoch: int = 1) -> Tuple[float, float, float, float]:
        self.model.train()
        total_losses: List[float] = []
        mse_losses: List[float] = []
        kl_losses: List[float] = []
        running_losses: deque[float] = deque(maxlen=20)
        reward_sum = 0.0
        reward_count = 0

        # Dispatch to sequential TBPTT if configured.
        if self._is_sru and self._temporal_cfg.sru_training_mode == "sequential_tbptt":
            return self._train_epoch_sequential_tbptt(epoch=epoch)
        if self._is_sru and self._temporal_cfg.sru_training_mode == "random_macroblock_tbptt":
            return self._train_epoch_macroblock_tbptt(epoch=epoch)

        # Update mask probability for this epoch (ramp schedule)
        if self._mask_cfg.enabled:
            self._current_mask_prob = current_mask_probability(
                epoch,
                self._mask_cfg.target_mask_probability,
                self._mask_cfg.ramp_epochs,
            )

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

        Causal mode::
            prev_actions[:, 0] = predecessor_action
            prev_actions[:, t] = actions[:, t-1]
            Transformer(obs[:, t], prev_actions[:, t]) → belief_t
            ControllerTrunk(belief_t, current_actions[:, t]) → reward_pred[t]
            MSE(reward_pred[t], rewards[:, t])

        SRU burn-in mode::
            The full sequence includes a burn-in prefix.
            valid_step=False marks padding (state unchanged, not part of episode).
            loss_mask=True marks only the 16 target positions.
            Gradients flow through the entire sequence (no detach at boundary).
            Reward MSE and KL are reduced only over target positions.
        """
        obs = batch["obs"].to(self.device, non_blocking=True)
        act = batch["action"].to(self.device, non_blocking=True)
        rew = batch["reward"].to(self.device, non_blocking=True)

        B, T_full = rew.shape

        if self._is_sru:
            # ---- SRU burn-in mode ----
            loss_mask = batch.get("loss_mask")
            if loss_mask is None:
                raise KeyError(
                    "SRU trainer requires loss_mask in batch. "
                    "Use RolloutDataset with recurrent_context=True."
                )
            loss_mask = loss_mask.to(self.device, non_blocking=True)
            valid_step = batch.get("valid_step")
            if valid_step is not None:
                valid_step = valid_step.to(self.device, non_blocking=True)

            # Previous actions over the full sequence.
            if "predecessor_action" not in batch:
                raise KeyError(
                    "Batch is missing required predecessor_action."
                )
            pred = batch["predecessor_action"].to(self.device, non_blocking=True)
            prev_actions = torch.zeros(B, T_full, ACTION_DIM, device=self.device)
            # Standard causal shift: prev_actions[:, t] = actions[:, t-1].
            if T_full > 1:
                prev_actions[:, 1:] = act[:, :T_full - 1]
            # Override the first VALID (non-padding) position with the correct
            # predecessor action.  Padded positions keep whatever prev_action
            # they have (state is unchanged by valid_step=False).
            if valid_step is not None:
                first_valid = valid_step.long().argmax(dim=1)  # (B,)
                for b in range(B):
                    fv = first_valid[b].item()
                    if valid_step[b, fv]:
                        prev_actions[b, fv] = pred[b]
            else:
                prev_actions[:, 0] = pred
            current_actions = act

            # Temporal observational dropout mask (D.1) — applied only to target positions.
            observation_keep = None
            if self._mask_cfg.enabled:
                # Mask only over the target window portion.
                target_T = min(self.sequence_len, T_full)
                obs_keep_target = sample_mask(
                    B, target_T,
                    warmup_steps=self._mask_cfg.warmup_steps,
                    horizons=self._mask_cfg.horizons,
                    mask_probability=self._current_mask_prob,
                    rng=self._train_rng,
                    device=self.device,
                )
                # Embed into full sequence: always visible in burn-in, masked in target.
                full_keep = torch.ones(B, T_full, device=self.device, dtype=torch.bool)
                target_start = T_full - target_T
                full_keep[:, target_start:] = obs_keep_target
                observation_keep = full_keep

            # Determine execution policy.
            exec_policy = getattr(self._mask_cfg, "observation_dropout_execution", "post_perception")

            out: WorldModelOutput = self.model.forward_sequence(
                obs, prev_actions, current_actions,
                force_keep_input=True,
                observation_keep=observation_keep,
                valid_step=valid_step,
                observation_dropout_execution=exec_policy,
            )

            r_pred_seq = out.reward_pred_seq
            r_true_seq = rew

            # Target-only reward MSE.
            loss_mask_f = loss_mask.float()  # (B, T)
            diff = (r_pred_seq - r_true_seq).pow(2)  # (B, T)
            loss_mse = (diff * loss_mask_f).sum() / loss_mask_f.sum().clamp_min(1)

            # Target-only KL.
            loss_kl = torch.zeros((), device=self.device)
            if out.tok_mu is not None and out.tok_logvar is not None:
                # In pre_perception_skip mode, KL must only apply to positions
                # that were actually perceived (observation_keep=True).
                kl_mask = loss_mask
                if exec_policy == "pre_perception_skip" and observation_keep is not None:
                    kl_mask = loss_mask & observation_keep
                if kl_mask.any():
                    loss_kl = masked_kl_normal(out.tok_mu, out.tok_logvar, kl_mask)

            return loss_mse + self.beta * loss_kl, loss_mse, loss_kl

        # ---- Causal mode (unchanged) ----
        assert self.sequence_len <= T_full

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

        observation_keep = None
        if self._mask_cfg.enabled:
            observation_keep = sample_mask(
                B, self.sequence_len,
                warmup_steps=self._mask_cfg.warmup_steps,
                horizons=self._mask_cfg.horizons,
                mask_probability=self._current_mask_prob,
                rng=self._train_rng,
                device=self.device,
            )

        out: WorldModelOutput = self.model.forward_sequence(
            obs[:, :self.sequence_len],
            prev_actions, current_actions,
            force_keep_input=True,
            observation_keep=observation_keep,
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
        total_target_count = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                obs = batch["obs"].to(self.device)
                act = batch["action"].to(self.device)
                rew = batch["reward"].to(self.device)

                B, T_full = rew.shape

                if self._is_sru:
                    loss_mask = batch.get("loss_mask")
                    if loss_mask is None:
                        raise KeyError(
                            "SRU evaluation requires loss_mask in batch."
                        )
                    loss_mask = loss_mask.to(self.device)
                    valid_step = batch.get("valid_step")
                    if valid_step is not None:
                        valid_step = valid_step.to(self.device)

                    pred = batch["predecessor_action"].to(self.device)
                    prev_actions = torch.zeros(B, T_full, ACTION_DIM, device=self.device)
                    if T_full > 1:
                        prev_actions[:, 1:] = act[:, :T_full - 1]
                    if valid_step is not None:
                        first_valid = valid_step.long().argmax(dim=1)
                        for b in range(B):
                            fv = first_valid[b].item()
                            if valid_step[b, fv]:
                                prev_actions[b, fv] = pred[b]
                    else:
                        prev_actions[:, 0] = pred
                    current_actions = act

                    out = self.model.forward_sequence(
                        obs, prev_actions, current_actions,
                        force_keep_input=True,
                        valid_step=valid_step,
                    )
                    r_pred = out.reward_pred_seq
                    r_true = rew

                    lm_f = loss_mask.float()  # (B, T)
                    n_target = lm_f.sum().item()
                    if n_target > 0:
                        diff = (r_pred - r_true).pow(2)  # (B, T)
                        total_mse += (diff * lm_f).sum().item()
                        total_abs_error += (torch.abs(r_pred - r_true) * lm_f).sum().item()
                        total_baseline_sse += (
                            (r_true - self._last_train_reward_mean).pow(2) * lm_f
                        ).sum().item()
                        total_count += n_target
                        total_target_count += int(n_target)
                else:
                    seq_len = min(self.sequence_len, T_full)

                    if "predecessor_action" not in batch:
                        raise KeyError(
                            "Batch is missing required predecessor_action."
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
        result = {
            "val_mse": mse,
            "val_mae": total_abs_error / max(1, total_count),
            "mean_baseline_mse": baseline_mse,
        }
        if self._is_sru:
            result["target_positions"] = total_target_count
            result["total_positions"] = total_count if not self._is_sru else 0
        return result

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def fit(
        self,
        validate_every: int = 1,
        progress_label: str = "Epoch",
        reference_targets_per_epoch: Optional[int] = None,
        reference_epochs: Optional[int] = None,
    ) -> Path:
        """Train for ``self.epochs`` dataset passes.

        The optional reference budget makes a nonstandard pass unit explicit
        without changing ordinary epoch-oriented training output.
        """
        if validate_every < 1:
            raise ValueError(f"validate_every must be >= 1, got {validate_every}")
        if reference_targets_per_epoch is not None and reference_targets_per_epoch < 1:
            raise ValueError("reference_targets_per_epoch must be >= 1 when provided")

        cumulative_targets = 0
        for epoch in range(1, self.epochs + 1):
            train_total, train_mse, train_kl, elapsed = self.train_one_epoch(epoch=epoch)

            is_val_epoch = (epoch % validate_every == 0) or (epoch == self.epochs)

            if is_val_epoch:
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
            else:
                row = {
                    "epoch": epoch,
                    "train_total": train_total,
                    "train_mse": train_mse,
                    "train_kl": train_kl,
                    "val_mse": float("nan"),
                    "val_mae": float("nan"),
                    "baseline_mse": float("nan"),
                    "time": elapsed,
                }
            gn = getattr(self, "_grad_norms", None)
            if gn:
                for k, vals in gn.items():
                    if vals:
                        row[f"gn_{k}"] = vals[-1]
            sm = getattr(self, "_seq_metrics", None)
            if sm:
                for k, v in sm.items():
                    row[k] = v
                cumulative_targets += int(sm.get("direct_supervised_targets", 0))


            progress_suffix = ""
            if reference_targets_per_epoch is not None:
                budget_epoch = cumulative_targets / reference_targets_per_epoch
                budget_total = reference_epochs if reference_epochs is not None else "?"
                progress_suffix = f" | budget epoch {budget_epoch:.2f}/{budget_total}"
            self.log_and_checkpoint(
                epoch, row, allow_best=is_val_epoch,
                progress_label=progress_label, progress_suffix=progress_suffix,
            )

        return self.out_dir / "best_world_model.pt"

    def log_and_checkpoint(
        self,
        epoch: int,
        row: Dict[str, float],
        allow_best: bool = True,
        progress_label: str = "Epoch",
        progress_suffix: str = "",
    ) -> None:
        write_header = not self.metrics_file.exists()
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self._global_step += 1

        if allow_best and "val_mse" in row:
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
                print(f"[{progress_label} {epoch}] skipped ckpt: val_mse={val_metric:.4f} best={self.best_val_metric:.4f}")
            print(f"[{progress_label} {epoch}] train_mse={row['train_mse']:.4f} train_kl={row['train_kl']:.4f} val_mse={row.get('val_mse', float('nan')):.4f} time={row['time']:.1f}s{progress_suffix}")
        else:
            # Non-validation pass: save latest checkpoint only.
            if self.config is not None:
                save_checkpoint(
                    path=self.out_dir / "checkpoint_latest",
                    model_state=self.model.state_dict(),
                    optimizer_state=self.optimizer.state_dict(),
                    config=self.config,
                    global_step=self._global_step,
                    epoch=epoch,
                    metrics={"train_mse": row["train_mse"]},
                    dataset_manifest_ref=self.dataset_manifest_ref,
                )
            print(f"[{progress_label} {epoch}] train_mse={row['train_mse']:.4f} train_kl={row['train_kl']:.4f} time={row['time']:.1f}s (no val){progress_suffix}")
