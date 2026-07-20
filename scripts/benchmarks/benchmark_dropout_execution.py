#!/usr/bin/env python3
"""Benchmark observational-dropout execution policies: post_perception vs pre_perception_skip.

Measures perceived-frame count, forward/backward time, and peak GPU memory
for the actual SRU context layout: 20 visible burn-in positions followed by a
16-position target window.  The mask curriculum applies only to that target
window.  It reports:
  - all-visible;
  - one deterministic batch matching the current p=0.5 curriculum in
    expectation (about 2.7 masked target frames/sample);
  - the strongest supported target-only blind block (12 masked target steps).

This is deliberately not an imagined-rollout benchmark: imagination already
avoids perception entirely and should be benchmarked through its own path.
"""

import argparse
import statistics
import time

import torch

from rwm.config.experiment_config import ExperimentConfig, TemporalConfig, TemporalMaskConfig, TrainingConfig
from rwm.models.rwm.model import ReducedWorldModel


def _make_batch(B: int, T: int, device: torch.device) -> dict:
    return {
        "obs": torch.randn(B, T, 3, 64, 64, device=device),
        "prev_actions": torch.randn(B, T, 3, device=device),
        "current_actions": torch.randn(B, T, 3, device=device),
        "valid_step": torch.ones(B, T, dtype=torch.bool, device=device),
    }


def _benchmark_loss(
    out, batch: dict, loss_mask: torch.Tensor,
    observation_keep: torch.Tensor, exec_policy: str,
) -> torch.Tensor:
    """Mirror the SRU trainer's target-only MSE/KL reductions."""
    mse = ((out.reward_pred_seq - batch["current_actions"][..., 0]).pow(2)
           * loss_mask.float()).sum() / loss_mask.sum().clamp_min(1)
    if out.tok_mu is None or out.tok_logvar is None:
        return mse
    kl_mask = loss_mask
    if exec_policy == "pre_perception_skip":
        kl_mask = loss_mask & observation_keep
    if not kl_mask.any():
        return mse
    kl = 0.5 * (out.tok_mu.pow(2) + out.tok_logvar.exp() - 1 - out.tok_logvar)
    return mse + 0.1 * kl[kl_mask].mean()


def _make_observation_keep(B: int, T: int, mode: str, device: torch.device) -> torch.Tensor:
    """Build masks with the same burn-in/target semantics used by training."""
    burn_in, target, warmup = 20, 16, 4
    if T != burn_in + target:
        raise ValueError(f"Expected SRU layout T={burn_in + target}, got T={T}")
    if mode == "all_visible":
        return torch.ones(B, T, dtype=torch.bool, device=device)
    keep = torch.ones(B, T, dtype=torch.bool, device=device)
    if mode == "curriculum_expected":
        # Four of eight samples receive one contiguous target block.  The
        # horizons sum to 21, close to p=0.5 * mean(1,2,4,8,12) * B = 21.6.
        for b, h in enumerate((1, 4, 8, 8)):
            if b < B:
                keep[b, burn_in + warmup:burn_in + warmup + h] = False
        return keep
    elif mode == "max_target_blind":
        # The curriculum's largest horizon: burn-in and target warmup remain
        # visible; the following 12 target positions are skipped.
        keep[:, burn_in + warmup:burn_in + target] = False
        return keep
    raise ValueError(f"Unknown mode: {mode}")


def bench(mode: str, exec_policy: str, device: torch.device, device_str: str,
          B: int = 8, T: int = 36, n_warmup: int = 10, n_iter: int = 30) -> dict:
    torch.manual_seed(0)

    cfg = ExperimentConfig(
        temporal=TemporalConfig(backend="minimal_sru", sru_burn_in_steps=20),
        training=TrainingConfig(temporal_mask=TemporalMaskConfig(
            enabled=True, warmup_steps=4, horizons=[1, 2, 4, 8, 12],
            observation_dropout_execution=exec_policy,
        )),
    )
    model = ReducedWorldModel(temporal_config=cfg.temporal).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    batch = _make_batch(B, T, device)
    observation_keep = _make_observation_keep(B, T, mode, device)
    # The 20 recurrent-context positions establish z; only the final 16
    # target positions receive direct reward/KL supervision.
    loss_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    loss_mask[:, 20:] = True

    # Warmup.
    for _ in range(n_warmup):
        optimizer.zero_grad()
        out = model.forward_sequence(
            batch["obs"], batch["prev_actions"], batch["current_actions"],
            force_keep_input=True, valid_step=batch["valid_step"],
            observation_keep=observation_keep,
            observation_dropout_execution=exec_policy,
        )
        loss = _benchmark_loss(out, batch, loss_mask, observation_keep, exec_policy)
        loss.backward()
        optimizer.step()

    before_frames = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0

    if device_str == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(n_iter):
        optimizer.zero_grad()
        out = model.forward_sequence(
            batch["obs"], batch["prev_actions"], batch["current_actions"],
            force_keep_input=True, valid_step=batch["valid_step"],
            observation_keep=observation_keep,
            observation_dropout_execution=exec_policy,
        )
        loss = _benchmark_loss(out, batch, loss_mask, observation_keep, exec_policy)
        loss.backward()
        optimizer.step()

    if device_str == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_gpu = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    n_visible = int(observation_keep.sum().item())

    return {
        "mode": mode,
        "exec_policy": exec_policy,
        "B": B, "T": T,
        "visible_frames": n_visible,
        "total_frames": B * T,
        "ms_per_call": elapsed / n_iter * 1000,
        "calls_per_sec": n_iter / elapsed,
        "peak_gpu_gb": peak_gpu,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trials", type=int, default=5,
                        help="Repeated, alternating-order trials per policy (default: 5).")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Timed forward/backward calls in each trial (default: 30).")
    args = parser.parse_args()
    device = torch.device(args.device)
    device_str = "cuda" if device.type == "cuda" else "cpu"

    results = []
    for mode in ["all_visible", "curriculum_expected", "max_target_blind"]:
        # Alternating execution order prevents a one-time GPU/cache warm-up
        # effect from being misreported as an execution-policy speedup.
        per_policy = {"post_perception": [], "pre_perception_skip": []}
        for trial in range(args.trials):
            policies = ["post_perception", "pre_perception_skip"]
            if trial % 2:
                policies.reverse()
            for exec_policy in policies:
                per_policy[exec_policy].append(
                    bench(mode, exec_policy, device, device_str, n_iter=args.iterations)
                )

        for exec_policy, trials in per_policy.items():
            # The median is robust to occasional scheduling spikes on the
            # display GPU.  The representative mask is intentionally fixed,
            # so the two policies process the same visible-frame count.
            ms_values = [r["ms_per_call"] for r in trials]
            mem_values = [r["peak_gpu_gb"] for r in trials]
            r = dict(trials[0])
            r["ms_per_call"] = statistics.median(ms_values)
            r["peak_gpu_gb"] = statistics.median(mem_values)
            r["calls_per_sec"] = 1000.0 / r["ms_per_call"]
            r["ms_min"] = min(ms_values)
            r["ms_max"] = max(ms_values)
            r["trials"] = args.trials
            results.append(r)

    print(f"\nDevice: {device_str}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"\n{'='*100}")
    print(f"{'Mode':<22} {'Policy':<22} {'Visible':<10} {'Total':<8} {'ms/call median':<17} {'range':<18} {'Peak GPU':<10}")
    print(f"{'-'*100}")
    for r in results:
        print(f"{r['mode']:<22} {r['exec_policy']:<22} {r['visible_frames']:<10} {r['total_frames']:<8} "
              f"{r['ms_per_call']:<17.4f} {r['ms_min']:.2f}–{r['ms_max']:.2f} ms{'':<4} "
              f"{r['peak_gpu_gb']:<10.4f}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
