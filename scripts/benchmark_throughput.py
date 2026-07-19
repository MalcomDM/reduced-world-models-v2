#!/usr/bin/env python3
"""Measure world-model training throughput before/after performance refactors.

Usage:
    python scripts/benchmark_throughput.py \
        --out runs/component_refinement/causal_transformer/benchmarks/baseline.json

Runs forward+backward on synthetic data with CUDA sync, reporting:
    windows/sec, frames/sec, ms/step, peak GPU mem, loader time.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np

from rwm.models.rwm.model import ReducedWorldModel
from rwm.config.config import ACTION_DIM


def benchmark(
    B: int = 8,
    T: int = 16,
    C: int = 3,
    H: int = 64,
    W: int = 64,
    warmup: int = 3,
    iterations: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    model = ReducedWorldModel(action_dim=ACTION_DIM).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    obs = torch.randn(B, T, C, H, W, device=device)
    prev_actions = torch.zeros(B, T, ACTION_DIM, device=device)
    if T > 1:
        actions = torch.randn(B, T, ACTION_DIM, device=device)
        prev_actions[:, 1:] = actions[:, :T - 1]
    else:
        actions = torch.zeros(B, T, ACTION_DIM, device=device)
    targets = torch.randn(B, T, device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        out = model.forward_sequence(obs, prev_actions, actions, force_keep_input=True)
        loss = torch.nn.functional.mse_loss(out.reward_pred_seq, targets)
        loss.backward()
        optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad()
        out = model.forward_sequence(obs, prev_actions, actions, force_keep_input=True)
        loss = torch.nn.functional.mse_loss(out.reward_pred_seq, targets)
        loss.backward()
        optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_steps = iterations * B * T
    peak_gpu = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    results = {
        "batch_size": B,
        "sequence_len": T,
        "device": device,
        "warmup_steps": warmup,
        "iterations": iterations,
        "total_frames": iterations * B * T,
        "total_steps": iterations,
        "elapsed_s": round(elapsed, 3),
        "ms_per_step": round(1000 * elapsed / iterations, 3),
        "windows_per_sec": round(iterations * B / elapsed, 1),
        "frames_per_sec": round(iterations * B * T / elapsed, 1),
        "peak_gpu_gb": round(peak_gpu, 4),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/component_refinement/causal_transformer/benchmarks/baseline.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-len", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    results = benchmark(
        B=args.batch_size,
        T=args.sequence_len,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
