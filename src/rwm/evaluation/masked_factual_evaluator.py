"""Masked-observation factual evaluator for Stage 2.5D.

Canonical semantics
-------------------
- Tokenizer evaluation mode ``mean`` (deterministic).
- Recurrent burn-in/context positions are always visible.
- ``warmup`` is counted from the first directly supervised position
  (``loss_mask=True``), not from layout position zero.
- Exactly ``mask_horizon`` subsequent valid target positions are masked and
  scored.
- Actions remain explicit: ``token[t]`` uses previous action ``a[t-1]``,
  reward prediction uses current action ``a[t]``.
- The factual reward target remains ``reward[t] = r[t+1]``.
- The model receives no image-derived information during the blind interval,
  but it may use recurrent/causal state and action history.

Usage
-----
::

    evaluator = MaskedFactualEvaluator(model, device, train_reward_mean=..., tokenizer_eval_mode="mean")
    loader = build_val_loader(...)
    results = evaluator.evaluate(loader, warmup=4, mask_horizons=(1, 2, 4, 8, 12))
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


def _make_prev_actions_correct(
    actions: torch.Tensor,
    predecessor_action: torch.Tensor,
    valid_step: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build prev_actions from factual data.

    ``prev_actions[:, 0] = predecessor_action``
    ``prev_actions[:, t] = actions[:, t-1]`` for ``t >= 1``.
    """
    B, T = actions.shape[0], actions.shape[1]
    prev = torch.empty_like(actions)
    prev[:, 0] = predecessor_action
    if T > 1:
        prev[:, 1:] = actions[:, :-1]
    if valid_step is not None:
        if valid_step.dtype != torch.bool or valid_step.shape != actions.shape[:2]:
            raise ValueError("valid_step must be bool with shape (B, T)")
        first_valid = valid_step.long().argmax(dim=1)
        for b in range(B):
            index = int(first_valid[b].item())
            if valid_step[b, index]:
                prev[b, index] = predecessor_action[b]
    return prev


def _make_prev_actions_zero(actions: torch.Tensor,
                             predecessor_action: torch.Tensor) -> torch.Tensor:
    """Zero previous actions."""
    return torch.zeros_like(actions)


def _make_prev_actions_shifted(actions: torch.Tensor,
                                predecessor_action: torch.Tensor) -> torch.Tensor:
    """One-step shifted: prev_actions[t] = actions[t] (current action used
    as previous action, which means the model sees action[t] as context
    when processing obs[t])."""
    return actions.clone()


def _make_observation_keep(T: int, warmup: int, mask_horizon: int,
                           device: torch.device) -> torch.Tensor:
    """Build a contiguous (1, T) factual blind horizon.

    The warmup is visible, exactly ``mask_horizon`` subsequent steps are
    masked, and later factual observations become visible again.  This lets
    each horizon measure a bounded period of prediction without looking.
    """
    keep = torch.ones(1, T, dtype=torch.bool, device=device)
    mask_end = min(warmup + mask_horizon, T)
    if mask_end > warmup:
        keep[:, warmup:mask_end] = False
    return keep


def _make_target_masks(
    *,
    loss_mask: Optional[torch.Tensor],
    valid_step: Optional[torch.Tensor],
    batch_size: int,
    sequence_len: int,
    warmup: int,
    mask_horizon: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build observation, scoring, and action-variant masks.

    ``loss_mask`` defines the directly supervised target region. When it is
    absent (the causal backend), every valid position is a target. Burn-in
    positions before the first target remain visible. ``warmup`` and
    ``mask_horizon`` count target positions, so an SRU layout with 20 burn-in
    positions and ``warmup=4`` starts masking at absolute position 24.
    """
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if mask_horizon < 1:
        raise ValueError("mask_horizon must be >= 1")

    if valid_step is None:
        valid = torch.ones(batch_size, sequence_len, dtype=torch.bool, device=device)
    else:
        valid = valid_step.to(device=device)
        if valid.dtype != torch.bool or valid.shape != (batch_size, sequence_len):
            raise ValueError("valid_step must be bool with shape (B, T)")

    if loss_mask is None:
        targets = valid.clone()
    else:
        targets = loss_mask.to(device=device)
        if targets.dtype != torch.bool or targets.shape != (batch_size, sequence_len):
            raise ValueError("loss_mask must be bool with shape (B, T)")
        targets = targets & valid

    observation_keep = valid.clone()
    score_mask = torch.zeros_like(valid)
    variant_mask = torch.zeros_like(valid)

    required = warmup + mask_horizon
    for b in range(batch_size):
        target_indices = targets[b].nonzero(as_tuple=True)[0]
        if target_indices.numel() < required:
            raise ValueError(
                "Not enough valid target positions for masked evaluation: "
                f"sample {b} has {target_indices.numel()}, requires {required} "
                f"(warmup={warmup}, horizon={mask_horizon})"
            )
        blind_indices = target_indices[warmup:required]
        observation_keep[b, blind_indices] = False
        score_mask[b, blind_indices] = True
        # Alter action history only inside the controlled blind interval.
        variant_mask[b, blind_indices] = True

    return observation_keep, score_mask, variant_mask


def _mean_std(values: List[float]) -> Tuple[float, float]:
    """Mean and standard deviation (population)."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, var ** 0.5


class MaskedFactualEvaluator:
    """Evaluator for masked-observation factual reward prediction.

    Parameters
    ----------
    model:
        ReducedWorldModel in eval mode.
    device:
        torch device.
    tokenizer_eval_mode:
        Tokenizer evaluation policy (``"mean"`` or ``"sample"``).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        train_reward_mean: float,
        tokenizer_eval_mode: str = "mean",
        observation_dropout_execution: str = "post_perception",
    ) -> None:
        self.model = model
        self.device = device
        if tokenizer_eval_mode != "mean":
            raise ValueError("Masked factual evaluation requires tokenizer_eval_mode='mean'")
        actual_mode = getattr(getattr(model, "tokenizer", None), "eval_mode", None)
        if actual_mode != "mean":
            raise ValueError(
                "Masked factual evaluation requires a model loaded with tokenizer mean mode; "
                f"got {actual_mode!r}"
            )
        self.train_reward_mean = float(train_reward_mean)
        self.tokenizer_eval_mode = "mean"
        if observation_dropout_execution not in {"post_perception", "pre_perception_skip"}:
            raise ValueError(
                "observation_dropout_execution must be 'post_perception' "
                "or 'pre_perception_skip'"
            )
        self.observation_dropout_execution = observation_dropout_execution
        model.eval()

    # ------------------------------------------------------------------
    # Action variant builders
    # ------------------------------------------------------------------

    ACTION_VARIANTS: Dict[str, Any] = {
        "correct": _make_prev_actions_correct,
        "zero": _make_prev_actions_zero,
        "shifted": _make_prev_actions_shifted,
    }

    # ------------------------------------------------------------------
    # Single-horizon evaluation
    # ------------------------------------------------------------------

    def evaluate_horizon(
        self,
        loader: DataLoader,
        warmup: int,
        mask_horizon: int,
        action_variant: str = "correct",
    ) -> Dict[str, Any]:
        """Evaluate one mask horizon with the given action variant.

        Returns dict with keys:
            warmup, mask_horizon, action_variant, transitions,
            val_mse, val_mae, baseline_mse, ratio,
            visible_ref_mse, delta_from_visible,
            observation_keep_sum (how many steps were visible),
            tokenizer_eval_mode.
        """
        assert action_variant in self.ACTION_VARIANTS, (
            f"Unknown action_variant {action_variant!r}; "
            f"choose from {list(self.ACTION_VARIANTS)}"
        )

        make_prev = self.ACTION_VARIANTS[action_variant]
        model = self.model
        device = self.device

        total_sse = 0.0
        total_abs = 0.0
        total_baseline_sse = 0.0
        visible_sse = 0.0
        visible_count = 0
        transition_count = 0
        valid_position_count = 0
        visible_valid_count = 0
        perceived_valid_count = 0
        window_count = 0

        with torch.no_grad():
            for batch in loader:
                obs = batch["obs"].to(device, non_blocking=True)
                actions = batch["action"].to(device, non_blocking=True)
                rewards = batch["reward"].to(device, non_blocking=True)
                predecessor = batch["predecessor_action"].to(device, non_blocking=True)

                B, T = obs.shape[0], obs.shape[1]
                valid_step = batch.get("valid_step")
                if valid_step is not None:
                    valid_step = valid_step.to(device, non_blocking=True)
                loss_mask = batch.get("loss_mask")
                if loss_mask is not None:
                    loss_mask = loss_mask.to(device, non_blocking=True)

                obs_keep, score_mask, variant_mask = _make_target_masks(
                    loss_mask=loss_mask,
                    valid_step=valid_step,
                    batch_size=B,
                    sequence_len=T,
                    warmup=warmup,
                    mask_horizon=mask_horizon,
                    device=device,
                )

                # Full-visible reference: run once per batch (shared across horizons)
                prev_correct = _make_prev_actions_correct(
                    actions, predecessor, valid_step=valid_step,
                )
                visible_keep = (
                    valid_step
                    if valid_step is not None
                    else torch.ones(B, T, dtype=torch.bool, device=device)
                )
                out_visible = model.forward_sequence(
                    obs, prev_correct, actions, force_keep_input=True,
                    observation_keep=visible_keep,
                    valid_step=valid_step,
                    observation_dropout_execution=self.observation_dropout_execution,
                )
                visible_preds = out_visible.reward_pred_seq  # (B, T)

                # Preserve factual action context during visible warmup.  The
                # alternative action history begins only at the blind boundary,
                # otherwise it would confound the warmup representation itself.
                prev_masked = prev_correct.clone()
                variant_prev = make_prev(actions, predecessor)
                prev_masked[variant_mask] = variant_prev[variant_mask]
                out_masked = model.forward_sequence(
                    obs, prev_masked, actions, force_keep_input=True,
                    observation_keep=obs_keep,
                    valid_step=valid_step,
                    observation_dropout_execution=self.observation_dropout_execution,
                )
                masked_preds = out_masked.reward_pred_seq  # (B, T)

                # Metrics: evaluate exactly the masked target positions.
                preds_masked = masked_preds[score_mask]
                targs_masked = rewards[score_mask]
                total_sse += F.mse_loss(preds_masked, targs_masked, reduction="sum").item()
                total_abs += torch.abs(preds_masked - targs_masked).sum().item()
                total_baseline_sse += (
                    (targs_masked - self.train_reward_mean).square().sum().item()
                )
                transition_count += preds_masked.numel()

                # Visible reference MSE on the identical positions.
                visible_sse += F.mse_loss(
                    visible_preds[score_mask], targs_masked, reduction="sum",
                ).item()
                visible_count += targs_masked.numel()
                valid = (
                    valid_step
                    if valid_step is not None
                    else torch.ones(B, T, dtype=torch.bool, device=device)
                )
                valid_position_count += int(valid.sum().item())
                visible_this = int((obs_keep & valid).sum().item())
                visible_valid_count += visible_this
                perceived_valid_count += (
                    visible_this
                    if self.observation_dropout_execution == "pre_perception_skip"
                    else int(valid.sum().item())
                )
                window_count += B

        if transition_count == 0:
            return {
                "warmup": warmup,
                "mask_horizon": mask_horizon,
                "action_variant": action_variant,
                "transitions": 0,
                "val_mse": float("nan"),
                "val_mae": float("nan"),
                "baseline_mse": float("nan"),
                "ratio": float("nan"),
                "visible_ref_mse": float("nan"),
                "delta_from_visible": float("nan"),
                "observation_keep_steps_per_window": 0,
                "visible_valid_positions": 0,
                "valid_input_positions": 0,
                "perceived_valid_positions": 0,
                "skipped_valid_positions": 0,
                "tokenizer_eval_mode": self.tokenizer_eval_mode,
                "observation_dropout_execution": self.observation_dropout_execution,
            }

        val_mse = total_sse / transition_count
        val_mae = total_abs / transition_count
        baseline_mse = total_baseline_sse / transition_count
        ratio = val_mse / max(1e-8, baseline_mse)
        visible_ref_mse = visible_sse / max(1, visible_count)
        delta = val_mse - visible_ref_mse

        return {
            "warmup": warmup,
            "mask_horizon": mask_horizon,
            "action_variant": action_variant,
            "transitions": transition_count,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "baseline_mse": baseline_mse,
            "ratio": ratio,
            "visible_ref_mse": visible_ref_mse,
            "delta_from_visible": delta,
            "observation_keep_steps_per_window": (
                visible_valid_count / max(1, window_count)
            ),
            "visible_valid_positions": visible_valid_count,
            "valid_input_positions": valid_position_count,
            "perceived_valid_positions": perceived_valid_count,
            "skipped_valid_positions": valid_position_count - perceived_valid_count,
            "tokenizer_eval_mode": self.tokenizer_eval_mode,
            "observation_dropout_execution": self.observation_dropout_execution,
        }

    # ------------------------------------------------------------------
    # Full evaluation across horizons and action variants
    # ------------------------------------------------------------------

    def evaluate(
        self,
        loader: DataLoader,
        warmup: int = 4,
        mask_horizons: Tuple[int, ...] = (1, 2, 4, 8, 12),
        action_variants: Tuple[str, ...] = ("correct", "zero", "shifted"),
    ) -> Dict[str, Any]:
        """Run masked factual evaluation across all horizons and action variants.

        Returns dict with:
            config: warmup, mask_horizons, action_variants
            horizons: list of per-horizon results (each from evaluate_horizon)
            summary: aggregated statistics
        """
        horizons = []
        for variant in action_variants:
            for horizon in mask_horizons:
                result = self.evaluate_horizon(
                    loader, warmup=warmup, mask_horizon=horizon,
                    action_variant=variant,
                )
                horizons.append(result)

        # Build summary
        summary = {
            "tokenizer_eval_mode": self.tokenizer_eval_mode,
            "observation_dropout_execution": self.observation_dropout_execution,
            "warmup": warmup,
            "mask_horizons": list(mask_horizons),
            "action_variants": list(action_variants),
            "horizons": horizons,
        }
        return summary
