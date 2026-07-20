"""Frozen-world-model imagined Actor-Critic training (Stage 5.x).

Architecture::

    observed warmup (4 frames) → z_t
    Actor(c_t) → a_t                     (differentiable or mode)
    RewardHead(z_t, a_t) → r_hat_{t+1}   (frozen, no_grad)
    advance(z_t, a_t) → z_{t+1}          (frozen, nongradient)

All world-model, ControllerTrunk, and reward-head parameters are frozen.
Only the Actor and Critic (online + target) are updated.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from rwm.config.config import ACTION_DIM, VALUES_DIM, SEQ_LEN, WORLD_STATE_DIM
from rwm.config.experiment_config import ActorCriticConfig as _ACConfig
from rwm.imagination import ImaginationRollout
from rwm.models.actor_critic import ActorCritic
from rwm.models.rwm.model import ReducedWorldModel


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ImaginedACTrainingConfig:
    """Hyperparameters for imagined Actor-Critic training."""
    warmup_steps: int = 4
    imagination_horizon: int = 4
    horizons: Optional[List[int]] = None  # curriculum; sampled per batch
    gamma: float = 0.997
    lambda_: float = 0.95
    entropy_coef: float = 1e-3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden_dim: int = 64
    target_update_rate: float = 0.01
    batch_size: int = 8
    max_batches: int = 10000
    log_every: int = 10
    checkpoint_every: int = 500

    def __post_init__(self) -> None:
        if self.horizons is not None:
            self.horizons = sorted(set(self.horizons))

    def validate(self) -> None:
        if not 1 <= self.imagination_horizon <= 12:
            raise ValueError(
                f"imagination_horizon must be in [1, 12], got "
                f"{self.imagination_horizon}"
            )
        if self.horizons is not None:
            for h in self.horizons:
                if not 1 <= h <= 12:
                    raise ValueError(
                        f"horizon in curriculum must be in [1, 12], got {h}"
                    )
        if self.warmup_steps < 1:
            raise ValueError("warmup_steps must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    @property
    def active_horizon_pool(self) -> List[int]:
        if self.horizons is not None:
            return self.horizons
        return [self.imagination_horizon]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImaginedACTrainingConfig":
        return cls(**data)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ImaginedACTrainer:
    """Frozen-world-model imagined Actor-Critic training loop.

    Parameters
    ----------
    model:
        Frozen ``ReducedWorldModel`` loaded from a checkpoint (must be in
        eval mode).
    train_loader:
        DataLoader yielding ``RolloutSample`` batches.
    train_cfg:
        Training hyperparameters.
    ac_cfg:
        Actor-Critic architecture hyperparameters.
    device:
        Torch device.
    out_dir:
        Directory for checkpoints, logs, and config.
    seed:
        Optional global random seed for reproducibility.
    probe_batch:
        Optional held-out batch for fixed-probe evaluation.  If provided,
        used by ``evaluate_fixed_probe()``.
    """

    def __init__(
        self,
        model: ReducedWorldModel,
        train_loader: DataLoader,
        train_cfg: Optional[ImaginedACTrainingConfig] = None,
        ac_cfg: Optional[_ACConfig] = None,
        device: Optional[torch.device] = None,
        out_dir: Optional[Path] = None,
        seed: Optional[int] = None,
        probe_batch: Optional[Dict[str, Tensor]] = None,
    ) -> None:
        if train_cfg is None:
            train_cfg = ImaginedACTrainingConfig()
        train_cfg.validate()
        self.cfg = train_cfg

        if seed is not None:
            torch.manual_seed(seed)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if out_dir is None:
            out_dir = Path("runs/imagined_ac")
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # World model — frozen, eval.
        self.model = model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Detect SRU backend.
        self._is_sru = getattr(self.model, "_temporal_backend", "causal_transformer") == "minimal_sru"

        # Modules.
        self.imag = ImaginationRollout(self.model).to(self.device)
        ac_full_cfg = _ACConfig(
            hidden_dim=train_cfg.hidden_dim,
            actor_lr=train_cfg.actor_lr,
            critic_lr=train_cfg.critic_lr,
            gamma=train_cfg.gamma,
            lambda_=train_cfg.lambda_,
            entropy_coef=train_cfg.entropy_coef,
            target_update_rate=train_cfg.target_update_rate,
        )
        self.ac = ActorCritic(self.model, ac_full_cfg).to(self.device)

        self.train_loader = train_loader
        self._data_iter = iter(train_loader)
        self._global_step = 0
        self._metrics_log: List[Dict[str, Any]] = []
        self._anchor_info: Dict[str, Optional[str]] = {"path": None, "hash": None}
        self._data_provenance: Dict[str, Any] = {}
        self._probe_batch = probe_batch
        self._step_rng = torch.Generator(device="cpu")
        if seed is not None:
            self._step_rng.manual_seed(seed + 42)

        # Config save.
        self._save_config()

    # ------------------------------------------------------------------
    # Warmup helper
    # ------------------------------------------------------------------

    def _prep_warmup(self, batch: Dict[str, Tensor], ws: Optional[int] = None
                     ) -> Tuple[Tensor, Tensor, Tensor]:
        """Extract warmup observations from a dataset batch.

        Causal path:
            Takes the first ``ws`` positions from the batch.

        SRU path:
            Takes all valid burn-in positions followed by the first ``ws``
            target positions — a contiguous window from position 0 through
            ``first_target + ws - 1`` (or ``T_total``, whichever is smaller).

            * Left-padding (``valid_step=False``) is included but the SRU
              holds ``z`` unchanged.
            * Every valid burn-in observation is visible.
            * ``predecessor_action`` is applied at the first valid position.
            * Each later position receives the preceding real action.
            * The first target position receives the final burn-in action.

        Returns ``(obs, prev_actions, current_actions)`` each shaped
        ``(B, T_warm, ...)`` where T_warm ≤ ``first_target + ws``.
        """
        if ws is None:
            ws = self.cfg.warmup_steps
        B = batch["obs"].shape[0]
        device = self.device

        if self._is_sru and "valid_step" in batch:
            valid_step = batch["valid_step"].to(device)
            loss_mask = batch.get("loss_mask")
            T_total = batch["obs"].shape[1]
            obs_all = batch["obs"].to(device, non_blocking=True)
            act_all = batch["action"].to(device, non_blocking=True)

            # First target position (loss_mask goes True for target).
            if loss_mask is not None:
                loss_mask_b = loss_mask.to(device)
                first_target = loss_mask_b.long().argmax(dim=1)  # (B,)
            else:
                first_target = valid_step.long().argmax(dim=1)
            first_valid = valid_step.long().argmax(dim=1)  # (B,)

            # T_warm: from position 0 through first_target + ws - 1 (clamped).
            T_warm = first_target + ws
            T_warm = T_total if T_warm.max() > T_total else T_warm  # keep per-sample

            # Handle variable per-sample T_warm (due to episode boundary).
            T_warm_v = int(T_warm.max().item()) if T_warm.numel() > 1 else int(T_warm.item())
            T_warm_v = min(T_warm_v, T_total)

            obs = obs_all[:, :T_warm_v]
            act_warm = act_all[:, :T_warm_v]

            pred = batch["predecessor_action"].to(device, non_blocking=True)

            prev_actions = torch.zeros(B, T_warm_v, ACTION_DIM, device=device)
            if T_warm_v > 1:
                prev_actions[:, 1:] = act_warm[:, :T_warm_v - 1]
            for b in range(B):
                fv = int(first_valid[b].item())
                if fv < T_warm_v:
                    prev_actions[b, fv] = pred[b]
            return obs, prev_actions, act_warm

        # Standard (causal): take first ws positions.
        obs = batch["obs"][:, :ws, :, :, :].to(device, non_blocking=True)
        act = batch["action"][:, :ws, :].to(device, non_blocking=True)
        pred = batch["predecessor_action"].to(device, non_blocking=True)

        prev_actions = torch.zeros(B, ws, ACTION_DIM, device=device)
        prev_actions[:, 0] = pred
        if ws > 1:
            prev_actions[:, 1:] = act[:, :ws - 1]

        return obs, prev_actions, act

    # ------------------------------------------------------------------
    # Trajectory generation
    # ------------------------------------------------------------------

    def generate_trajectory(
        self,
        warmup_state: "ImaginationState",
        horizon: Optional[int] = None,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate an imagined trajectory.

        Parameters
        ----------
        warmup_state:
            Result of a warmup call.
        horizon:
            Number of imagined steps (defaults to config).
        deterministic:
            If True, use ``mode()`` instead of ``rsample()``.

        Returns
        -------
        states:
            ``(B, H, D)`` — beliefs ``z_t`` before each action.
        actions:
            ``(B, H, A)`` — sampled or deterministic actions.
        rewards:
            ``(B, H)`` — frozen reward-head predictions.
        z_H:
            ``(B, D)`` — belief after the final advance.
        """
        H = horizon if horizon is not None else self.cfg.imagination_horizon
        device = self.device
        B = warmup_state.current_belief.shape[0]
        D = warmup_state.current_belief.shape[-1]

        z_t = warmup_state.current_belief
        history = warmup_state.history
        lengths = warmup_state.lengths
        curr_z = z_t if warmup_state.is_sru else None

        states = torch.empty(B, H, D, device=device)
        actions = torch.empty(B, H, ACTION_DIM, device=device)
        rewards = torch.empty(B, H, device=device)

        for h in range(H):
            c_t = self.ac.encode(z_t)
            dist = self.ac.actor(c_t)
            a_t = dist.mode() if deterministic else dist.rsample()

            with torch.no_grad():
                _, r_t = self.model.controller(z_t, a_t)

            states[:, h] = z_t
            actions[:, h] = a_t
            rewards[:, h] = r_t.squeeze(-1)

            history, lengths, z_t = self.imag.advance(
                history, lengths, a_t, temporal_state=curr_z,
            )
            if curr_z is not None:
                curr_z = z_t  # SRU: z_t IS the next recurrent state

        return states, actions, rewards, z_t

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Single training step over one batch.

        1. Optionally sample horizon from curriculum.
        2. Warmup from observed frames.
        3. Generate ``H`` imagined steps via Actor.
        4. Compute bootstrap value from target Critic.
        5. Call ``ActorCritic.optimizer_step``.

        Returns metrics dict.
        """
        ws = self.cfg.warmup_steps
        pool = self.cfg.active_horizon_pool

        if len(pool) == 1:
            H = pool[0]
        else:
            idx = torch.randint(len(pool), (), generator=self._step_rng).item()
            H = pool[idx]

        device = self.device

        obs, prev_actions, current_actions = self._prep_warmup(batch)
        vs_warm = (batch["valid_step"][:, :obs.shape[1]].to(self.device)
                   if self._is_sru and "valid_step" in batch else None)
        warmup_state = self.imag.warmup(
            obs, prev_actions, current_actions, force_keep_input=True,
            valid_step=vs_warm,
        )

        states, actions_t, rewards_t, z_H = self.generate_trajectory(
            warmup_state, horizon=H,
        )

        with torch.no_grad():
            c_H = self.ac.encode(z_H)
            bootstrap_value = self.ac.target_critic(c_H).squeeze(-1)

        B_ = states.shape[0]
        terminated = torch.zeros(B_, H, dtype=torch.bool, device=device)
        continuation = torch.ones(B_, H, dtype=torch.bool, device=device)

        metrics = self.ac.optimizer_step(
            states, actions_t, rewards_t,
            terminated, continuation, bootstrap_value,
            gamma=self.cfg.gamma,
            lambda_=self.cfg.lambda_,
            entropy_coef=self.cfg.entropy_coef,
        )

        # Extra tracked values.
        with torch.no_grad():
            metrics["horizon"] = H
            metrics["batch_size"] = B_
            metrics["imagined_reward_mean"] = rewards_t.mean().item()

            # Sampled action statistics.
            steer_s = actions_t[..., 0]
            gas_s = actions_t[..., 1]
            brake_s = actions_t[..., 2]
            metrics["steer_mean"] = steer_s.mean().item()
            metrics["steer_std"] = steer_s.std().item()
            metrics["gas_mean"] = gas_s.mean().item()
            metrics["gas_std"] = gas_s.std().item()
            metrics["brake_mean"] = brake_s.mean().item()
            metrics["brake_std"] = brake_s.std().item()

            # Logstd and mode from the current policy over these states.
            Ds = states.shape[-1]
            flat_c = self.ac.encode(states.reshape(-1, Ds))
            dist = self.ac.actor(flat_c)
            logstd = dist.logstd.reshape(B_, H, ACTION_DIM)
            mode_a = dist.mode().reshape(B_, H, ACTION_DIM)

            metrics["steer_logstd_mean"] = logstd[..., 0].mean().item()
            metrics["gas_logstd_mean"] = logstd[..., 1].mean().item()
            metrics["brake_logstd_mean"] = logstd[..., 2].mean().item()
            metrics["steer_mode_mean"] = mode_a[..., 0].mean().item()
            metrics["gas_mode_mean"] = mode_a[..., 1].mean().item()
            metrics["brake_mode_mean"] = mode_a[..., 2].mean().item()

            # Near-bound rates (deterministic mode).
            sm = mode_a[..., 0]
            gm = mode_a[..., 1]
            bm = mode_a[..., 2]
            metrics["steer_sat_0.90"] = (sm.abs() > 0.90).float().mean().item()
            metrics["steer_sat_0.98"] = (sm.abs() > 0.98).float().mean().item()
            metrics["gas_sat_0.90"] = (gm > 0.90).float().mean().item()
            metrics["gas_sat_0.98"] = (gm > 0.98).float().mean().item()
            metrics["brake_sat_0.90"] = (bm > 0.90).float().mean().item()
            metrics["brake_sat_0.98"] = (bm > 0.98).float().mean().item()

        return metrics

    # ------------------------------------------------------------------
    # Fixed-probe evaluation
    # ------------------------------------------------------------------

    def evaluate_fixed_probe(self, horizons: Sequence[int] = (1, 2, 4)
                             ) -> Dict[str, Any]:
        """Deterministic probe using ``mode()`` on a held-out batch.

        Uses ``self._probe_batch`` (set at construction).  This is an offline
        imagined diagnostic only — it does not involve the real environment.

        Returns a dict keyed by horizon with sub-dicts containing:
            ``imagined_return``, ``value_mean``,
            ``steer_mean``, ``steer_std``, ``steer_saturation``,
            ``gas_mean``, ``gas_std``, ``gas_saturation``,
            ``brake_mean``, ``brake_std``, ``brake_saturation``.
        """
        if self._probe_batch is None:
            return {"error": "no probe batch available"}

        batch = {k: v.to(self.device, non_blocking=True)
                 for k, v in self._probe_batch.items()}
        results: Dict[str, Any] = {}

        for H in horizons:
            ws = self.cfg.warmup_steps
            obs, prev_actions, current_actions = self._prep_warmup(batch, ws=ws)
            vs_warm = (batch["valid_step"][:, :obs.shape[1]].to(self.device)
                       if self._is_sru and "valid_step" in batch else None)
            warmup_state = self.imag.warmup(
                obs, prev_actions, current_actions, force_keep_input=True,
                valid_step=vs_warm,
            )

            states, actions_t, rewards_t, z_H = self.generate_trajectory(
                warmup_state, horizon=H, deterministic=True,
            )

            c_H = self.ac.encode(z_H)
            with torch.no_grad():
                bootstrap_value = self.ac.target_critic(c_H).squeeze(-1)

            imagined_return = rewards_t.sum(dim=-1).mean().item()
            value_mean = self.ac.critic(
                self.ac.encode(states.reshape(-1, states.shape[-1]))
            ).reshape(-1, H).mean().item()

            steer = actions_t[..., 0]
            gas = actions_t[..., 1]
            brake = actions_t[..., 2]

            # Logstd from the policy at probe states.
            Ds = states.shape[-1]
            flat_c = self.ac.encode(states.reshape(-1, Ds))
            probe_dist = self.ac.actor(flat_c)
            ls = probe_dist.logstd.reshape(-1, H, ACTION_DIM)

            results[str(H)] = {
                "imagined_return": imagined_return,
                "value_mean": value_mean,
                # Sampled-action stats (mode actions since deterministic=True).
                "steer_mean": steer.mean().item(),
                "steer_std": steer.std().item(),
                "gas_mean": gas.mean().item(),
                "gas_std": gas.std().item(),
                "brake_mean": brake.mean().item(),
                "brake_std": brake.std().item(),
                # Logstd.
                "steer_logstd_mean": ls[..., 0].mean().item(),
                "gas_logstd_mean": ls[..., 1].mean().item(),
                "brake_logstd_mean": ls[..., 2].mean().item(),
                # Near-bound rates (deterministic mode).
                "steer_sat_0.90": (steer.abs() > 0.90).float().mean().item(),
                "steer_sat_0.98": (steer.abs() > 0.98).float().mean().item(),
                "gas_sat_0.90": (gas > 0.90).float().mean().item(),
                "gas_sat_0.98": (gas > 0.98).float().mean().item(),
                "brake_sat_0.90": (brake > 0.90).float().mean().item(),
                "brake_sat_0.98": (brake > 0.98).float().mean().item(),
            }

        return results

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self, num_batches: Optional[int] = None) -> None:
        """Run the training loop.

        Parameters
        ----------
        num_batches:
            Override for ``cfg.max_batches`` (used for smoke tests).
        """
        if num_batches is None:
            num_batches = self.cfg.max_batches

        self._metrics_log = []
        start_time = time.time()

        for step in range(1, num_batches + 1):
            self._global_step += 1

            try:
                batch = next(self._data_iter)
            except StopIteration:
                self._data_iter = iter(self.train_loader)
                batch = next(self._data_iter)

            batch = {k: v for k, v in batch.items()}

            metrics = self.train_step(batch)
            metrics["step"] = self._global_step
            self._metrics_log.append(metrics)

            if step % self.cfg.log_every == 0 or step == 1:
                elapsed = time.time() - start_time
                self._log_step(step, metrics, elapsed)

            if step % self.cfg.checkpoint_every == 0:
                self._save_checkpoint(step)

        self._save_checkpoint(num_batches)
        self._write_metrics_csv()
        self._write_training_summary(time.time() - start_time)
        if not (self.out_dir / "anchor_checkpoint.txt").exists():
            self._save_anchor_info()

    # ------------------------------------------------------------------
    # Logging / persistence
    # ------------------------------------------------------------------

    def _log_step(self, step: int, m: Dict[str, float], elapsed: float) -> None:
        H = m.get("horizon", "?")
        items = " | ".join(
            f"{k}={v:.4f}" for k, v in m.items()
            if k not in ("step", "horizon")
        )
        print(f"[{step:5d}] H={H} {items}  |  {elapsed:.1f}s")

    def _save_config(self) -> None:
        cfg_path = self.out_dir / "imagined_ac_config.json"
        with open(cfg_path, "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2, sort_keys=True)

    def _write_metrics_csv(self) -> None:
        if not self._metrics_log:
            return
        path = self.out_dir / "metrics.csv"
        keys = list(self._metrics_log[0].keys())
        with open(path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in self._metrics_log:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

    def _write_training_summary(self, elapsed_s: float) -> None:
        """Persist the optimisation budget without conflating it with data."""
        horizon_counts: Dict[str, int] = {}
        imagined_transitions = 0
        warmup_transitions = 0
        for row in self._metrics_log:
            horizon = int(row["horizon"])
            batch_size = int(row["batch_size"])
            horizon_counts[str(horizon)] = horizon_counts.get(str(horizon), 0) + 1
            imagined_transitions += horizon * batch_size
            warmup_transitions += self.cfg.warmup_steps * batch_size

        peak_gpu_gb: Optional[float] = None
        if self.device.type == "cuda":
            peak_gpu_gb = torch.cuda.max_memory_allocated(self.device) / 1e9
        with open(self.out_dir / "training_summary.json", "w") as f:
            json.dump({
                "actor_critic_updates": len(self._metrics_log),
                "horizon_update_counts": horizon_counts,
                "imagined_transitions": imagined_transitions,
                "factual_warmup_transitions_reused": warmup_transitions,
                "elapsed_s": elapsed_s,
                "peak_gpu_gb": peak_gpu_gb,
                "data_provenance": self._data_provenance,
            }, f, indent=2, sort_keys=True)

    def _save_checkpoint(self, step: int) -> None:
        checkpoint_dir = self.out_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        ckpt_path = checkpoint_dir / f"ac_checkpoint_{step}.pt"
        ac_state = {
            "actor": self.ac.actor.state_dict(),
            "critic": self.ac.critic.state_dict(),
            "target_critic": self.ac.target_critic.state_dict(),
        }
        opt_state = {
            "actor_optim": self.ac._actor_optim.state_dict(),
            "critic_optim": self.ac._critic_optim.state_dict(),
        }
        torch.save({
            "schema_version": 1,
            "kind": "imagined_actor_critic",
            "step": step,
            "global_step": self._global_step,
            "actor_critic": ac_state,
            "optimizer": opt_state,
            "config": self.cfg.to_dict(),
            "actor_critic_config": dataclasses.asdict(self.ac.cfg),
            "anchor": self._anchor_info,
            "data_provenance": self._data_provenance,
        }, ckpt_path)

    def load_actor_critic_checkpoint(self, path: Path) -> int:
        data = torch.load(path, map_location=self.device, weights_only=False)
        if data.get("kind") != "imagined_actor_critic":
            raise ValueError(f"not an imagined Actor-Critic checkpoint: {path}")
        state = data["actor_critic"]
        self.ac.actor.load_state_dict(state["actor"])
        self.ac.critic.load_state_dict(state["critic"])
        self.ac.target_critic.load_state_dict(state["target_critic"])
        self.ac._actor_optim.load_state_dict(data["optimizer"]["actor_optim"])
        self.ac._critic_optim.load_state_dict(data["optimizer"]["critic_optim"])
        self._global_step = int(data.get("global_step", data["step"]))
        self._anchor_info = data.get("anchor", self._anchor_info)
        self._data_provenance = data.get("data_provenance", self._data_provenance)
        return self._global_step

    def _save_anchor_info(self) -> None:
        path = self.out_dir / "anchor_checkpoint.txt"
        with open(path, "w") as f:
            f.write("Anchor checkpoint not recorded (set via CLI)\n")

    def set_anchor_info(self, ckpt_path: str, ckpt_hash: str) -> None:
        self._anchor_info = {"path": ckpt_path, "hash": ckpt_hash}
        path = self.out_dir / "anchor_checkpoint.txt"
        with open(path, "w") as f:
            f.write(f"path: {ckpt_path}\nhash: {ckpt_hash}\n")

    def set_data_provenance(self, provenance: Dict[str, Any]) -> None:
        """Attach the exact factual file split used to construct dream starts."""
        self._data_provenance = dict(provenance)


def load_ac_from_checkpoint(
    model: ReducedWorldModel,
    ac_checkpoint_path: Path,
    device: torch.device,
    actor_critic_config: Optional[_ACConfig] = None,
) -> "ActorCritic":
    """Load an Actor-Critic checkpoint into a fresh AC module without side effects.

    Unlike ``ImaginedACTrainer.load_actor_critic_checkpoint``, this function
    does not create any output directory or require a DataLoader.
    """
    from rwm.models.actor_critic import ActorCritic as _ActorCritic
    from rwm.config.experiment_config import ActorCriticConfig as _ACConfigDefault

    cfg = actor_critic_config or _ACConfigDefault()
    ac = _ActorCritic(model, cfg).to(device)
    data = torch.load(ac_checkpoint_path, map_location=device, weights_only=False)
    if data.get("kind") != "imagined_actor_critic":
        raise ValueError(f"not an imagined Actor-Critic checkpoint: {ac_checkpoint_path}")
    ac.actor.load_state_dict(data["actor_critic"]["actor"])
    ac.critic.load_state_dict(data["actor_critic"]["critic"])
    ac.target_critic.load_state_dict(data["actor_critic"]["target_critic"])
    if "optimizer" in data:
        if "actor_optim" in data["optimizer"]:
            ac._actor_optim.load_state_dict(data["optimizer"]["actor_optim"])
        if "critic_optim" in data["optimizer"]:
            ac._critic_optim.load_state_dict(data["optimizer"]["critic_optim"])
    return ac


def validate_ac_anchor_checkpoint(
    ac_checkpoint_path: Path,
    expected_anchor_hash: str,
) -> None:
    """Reject Actor-Critic state trained from a different world-model anchor."""
    data = torch.load(ac_checkpoint_path, map_location="cpu", weights_only=False)
    if data.get("kind") != "imagined_actor_critic":
        raise ValueError(
            f"not an imagined Actor-Critic checkpoint: {ac_checkpoint_path}"
        )
    recorded = data.get("anchor", {}).get("hash")
    if not recorded:
        raise ValueError(
            "Stage 6.1 requires Actor-Critic checkpoint anchor provenance"
        )
    if recorded != expected_anchor_hash:
        raise ValueError(
            "Actor-Critic/world-model anchor mismatch: "
            f"AC records {recorded}, selected anchor is {expected_anchor_hash}"
        )
