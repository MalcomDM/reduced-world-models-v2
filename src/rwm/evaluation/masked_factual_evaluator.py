"""Masked-observation factual evaluator for Stage 2.5D.

Canonical semantics
-------------------
- Tokenizer evaluation mode ``mean`` (deterministic).
- Warmup observations are visible.
- From a specified mask start onward, spatial observation representation
  is replaced with zeros.
- Actions remain explicit: ``token[t]`` uses previous action ``a[t-1]``,
  reward prediction uses current action ``a[t]``.
- The factual reward target remains ``reward[t] = r[t+1]``.
- The model receives no image-derived information after mask start, but
  it may use causal history and action history.

Usage
-----
::

    evaluator = MaskedFactualEvaluator(model, device, train_reward_mean=..., tokenizer_eval_mode="mean")
    loader = build_val_loader(...)
    results = evaluator.evaluate(loader, warmup=4, mask_horizons=(1, 2, 4, 8, 16))
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


def _make_prev_actions_correct(actions: torch.Tensor,
                                predecessor_action: torch.Tensor) -> torch.Tensor:
    """Build prev_actions from factual data.

    ``prev_actions[:, 0] = predecessor_action``
    ``prev_actions[:, t] = actions[:, t-1]`` for ``t >= 1``.
    """
    B, T = actions.shape[0], actions.shape[1]
    prev = torch.empty_like(actions)
    prev[:, 0] = predecessor_action
    if T > 1:
        prev[:, 1:] = actions[:, :-1]
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

        with torch.no_grad():
            for batch in loader:
                obs = batch["obs"].to(device, non_blocking=True)
                actions = batch["action"].to(device, non_blocking=True)
                rewards = batch["reward"].to(device, non_blocking=True)
                predecessor = batch["predecessor_action"].to(device, non_blocking=True)

                B, T = obs.shape[0], obs.shape[1]

                # Full-visible reference: run once per batch (shared across horizons)
                prev_correct = _make_prev_actions_correct(actions, predecessor)
                out_visible = model.forward_sequence(
                    obs, prev_correct, actions, force_keep_input=True,
                )
                visible_preds = out_visible.reward_pred_seq  # (B, T)

                # Preserve factual action context during visible warmup.  The
                # alternative action history begins only at the blind boundary,
                # otherwise it would confound the warmup representation itself.
                prev_masked = prev_correct.clone()
                if warmup < T:
                    variant_prev = make_prev(actions, predecessor)
                    prev_masked[:, warmup:] = variant_prev[:, warmup:]
                obs_keep = _make_observation_keep(T, warmup, mask_horizon, device)
                out_masked = model.forward_sequence(
                    obs, prev_masked, actions, force_keep_input=True,
                    observation_keep=obs_keep.expand(B, -1),
                )
                masked_preds = out_masked.reward_pred_seq  # (B, T)

                # Count visible transitions (warmup steps only visible in masked mode)
                visible_steps = min(warmup, T)

                # Metrics: evaluate only masked positions
                masked_start = warmup
                masked_end = min(warmup + mask_horizon, T)
                if masked_end > masked_start:
                    preds_masked = masked_preds[:, masked_start:masked_end]
                    targs_masked = rewards[:, masked_start:masked_end]
                    total_sse += F.mse_loss(preds_masked, targs_masked, reduction="sum").item()
                    total_abs += torch.abs(preds_masked - targs_masked).sum().item()
                    total_baseline_sse += (
                        (targs_masked - self.train_reward_mean).square().sum().item()
                    )
                    transition_count += preds_masked.numel()

                    # Visible reference MSE on the same positions
                    visible_sse += F.mse_loss(
                        visible_preds[:, masked_start:masked_end], targs_masked,
                        reduction="sum",
                    ).item()
                    visible_count += targs_masked.numel()

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
                "tokenizer_eval_mode": self.tokenizer_eval_mode,
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
            "observation_keep_steps_per_window": T - (masked_end - masked_start),
            "tokenizer_eval_mode": self.tokenizer_eval_mode,
        }

    # ------------------------------------------------------------------
    # Full evaluation across horizons and action variants
    # ------------------------------------------------------------------

    def evaluate(
        self,
        loader: DataLoader,
        warmup: int = 4,
        mask_horizons: Tuple[int, ...] = (1, 2, 4, 8, 16),
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
            "warmup": warmup,
            "mask_horizons": list(mask_horizons),
            "action_variants": list(action_variants),
            "horizons": horizons,
        }
        return summary
