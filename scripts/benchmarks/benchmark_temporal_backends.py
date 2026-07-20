#!/usr/bin/env python3
"""Reproducible temporal-only benchmark: causal Transformer vs MinimalSRUTemporal.

Measures:
  - forward-only inference (train-sequence, incremental, blind rollout)
  - forward+backward (train-sequence)
  - CPU and CUDA separately (use --device)

Causal Transformer uses a steady rolling history of SEQ_LEN=20 tokens.
MinimalSRUTemporal uses step() for incremental and forward_sequence() for
batched.  Inputs are pre-generated outside the timed region.

Usage:
    python scripts/benchmarks/benchmark_temporal_backends.py [--device cpu|cuda]
"""

import argparse
import time

import torch
import torch.nn as nn

from rwm.config.config import SEQ_LEN, VALUES_DIM, ACTION_DIM, WORLD_STATE_DIM
from rwm.models.rwm.causal_transformer import CausalTransformer
from rwm.models.rwm.minimal_sru_temporal import MinimalSRUTemporal


# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------

INPUT_DIM = VALUES_DIM + ACTION_DIM  # 35 for causal
SRU_INPUT_DIM = VALUES_DIM + ACTION_DIM + 1  # 36 for SRU


def _make_causal_batch(B: int, T: int, device: torch.device) -> torch.Tensor:
    """(B, T, 35) tokens."""
    return torch.randn(B, T, INPUT_DIM, device=device)


def _make_sru_batch(B: int, T: int, device: torch.device) -> torch.Tensor:
    """(B, T, 36) tokens."""
    return torch.randn(B, T, SRU_INPUT_DIM, device=device)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _bench_causal_forward_sequence(
    model: nn.Module, x: torch.Tensor, n_iter: int, device_str: str,
) -> float:
    """CausalTransformer forward with return_all=True."""
    for _ in range(n_iter):
        _ = model(x, return_all=True)
        if device_str == "cuda":
            torch.cuda.synchronize()


def _bench_sru_forward_sequence(
    model: nn.Module, x: torch.Tensor, n_iter: int, device_str: str,
) -> float:
    """MinimalSRUTemporal forward_sequence."""
    for _ in range(n_iter):
        _, _ = model.forward_sequence(x, return_all=True)
        if device_str == "cuda":
            torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Timed benchmark runners
# ---------------------------------------------------------------------------


def bench_train_sequence(
    label: str, params: int, is_sru: bool,
    B: int, T: int, device: torch.device, device_str: str,
    n_warmup: int = 10, n_iter: int = 100,
) -> dict:
    """Forward + backward (train mode)."""
    if is_sru:
        model = MinimalSRUTemporal(carry_bias_init=1.0).to(device).train()
        x = _make_sru_batch(B, T, device)
    else:
        model = CausalTransformer().to(device).train()
        x = _make_causal_batch(B, T, device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    def step():
        optimizer.zero_grad()
        if is_sru:
            out, _ = model.forward_sequence(x, return_all=True)
        else:
            out = model(x, return_all=True)
        out.sum().backward()
        optimizer.step()
        if device_str == "cuda":
            torch.cuda.synchronize()

    for _ in range(n_warmup):
        step()

    start = time.perf_counter()
    for _ in range(n_iter):
        step()
    elapsed = time.perf_counter() - start

    ms = elapsed / n_iter * 1000
    return {
        "label": label, "mode": "train_sequence",
        "B": B, "T": T, "params": params,
        "ms_per_call": ms, "calls_per_sec": n_iter / elapsed,
    }


def bench_incremental(
    label: str, params: int, is_sru: bool,
    B: int, device: torch.device, device_str: str,
    n_warmup: int = 20, n_iter: int = 500,
) -> dict:
    """Single-step forward only (eval mode).

    Causal: maintain rolling history of SEQ_LEN=20; append 1 token,
    left-truncate, call forward.  Includes history maintenance.
    SRU: single step() call.
    """
    if is_sru:
        model = MinimalSRUTemporal(carry_bias_init=1.0).to(device).eval()
        x_step = _make_sru_batch(B, 1, device)
    else:
        model = CausalTransformer().to(device).eval()
        # Pre-built rolling history of SEQ_LEN tokens.
        hist = _make_causal_batch(B, SEQ_LEN, device)
        new_token = _make_causal_batch(B, 1, device)

    with torch.no_grad():
        for _ in range(n_warmup):
            if is_sru:
                model.step(x_step.squeeze(1))
            else:
                hist = torch.cat([hist[:, 1:, :], new_token], dim=1)
                _ = model(hist)
            if device_str == "cuda":
                torch.cuda.synchronize()

    if device_str == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(n_iter):
            if is_sru:
                model.step(x_step.squeeze(1))
            else:
                hist = torch.cat([hist[:, 1:, :], new_token], dim=1)
                _ = model(hist)
            if device_str == "cuda":
                torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    ms = elapsed / n_iter * 1000
    return {
        "label": label, "mode": "incremental",
        "B": B, "params": params,
        "ms_per_call": ms, "calls_per_sec": n_iter / elapsed,
    }


def bench_blind_rollout_sru(
    label: str, params: int,
    B: int, H: int, device: torch.device, device_str: str,
    n_warmup: int = 10, n_iter: int = 100,
) -> dict:
    """SRU blind H-step rollout.

    Warm up with T_warm=4 factual tokens (forward_sequence once),
    then H blind steps each using step() with a single [zeros, action, keep=0] token.
    """
    model = MinimalSRUTemporal(carry_bias_init=1.0).to(device).eval()
    T_warm = 4

    # Pre-generated inputs.
    x_warm = _make_sru_batch(B, T_warm, device)
    x_blind = _make_sru_batch(B, 1, device)  # single blind step per call

    with torch.no_grad():
        for _ in range(n_warmup):
            _, z = model.forward_sequence(x_warm, return_all=True)
            for _ in range(H):
                z = model.step(x_blind.squeeze(1), z_prev=z)
            if device_str == "cuda":
                torch.cuda.synchronize()

    if device_str == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(n_iter):
            _, z = model.forward_sequence(x_warm, return_all=True)
            for _ in range(H):
                z = model.step(x_blind.squeeze(1), z_prev=z)
            if device_str == "cuda":
                torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    ms = elapsed / n_iter * 1000
    return {
        "label": label, "mode": "blind_rollout_sru",
        "B": B, "H": H, "params": params,
        "ms_per_call": ms, "calls_per_sec": n_iter / elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark temporal backends")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    device_str = "cuda" if device.type == "cuda" else "cpu"
    print(f"Device: {device_str}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # Parameters.
    causal_params = sum(p.numel() for p in CausalTransformer().parameters())
    sru = MinimalSRUTemporal(carry_bias_init=1.0)
    sru_params = sum(p.numel() for p in sru.parameters())

    print(f"\nParameters: CausalTransformer={causal_params}, MinimalSRUTemporal={sru_params}\n")

    results = []

    # --- Train sequence (forward + backward) ---
    for is_sru, label, params in [
        (False, f"CausalTransformer", causal_params),
        (True, f"MinimalSRUTemporal", sru_params),
    ]:
        r = bench_train_sequence(
            label, params, is_sru,
            B=32, T=20, device=device, device_str=device_str,
        )
        results.append(r)

    # --- Forward-only incremental ---
    for is_sru, label, params in [
        (False, f"CausalTransformer", causal_params),
        (True, f"MinimalSRUTemporal", sru_params),
    ]:
        r = bench_incremental(
            label, params, is_sru,
            B=1, device=device, device_str=device_str,
        )
        results.append(r)

    # --- SRU blind H=12 rollout ---
    if device.type == "cuda":
        r = bench_blind_rollout_sru(
            "MinimalSRUTemporal", sru_params,
            B=1, H=12, device=device, device_str=device_str,
        )
        results.append(r)

    # --- Summary ---
    print(f"{'=' * 110}")
    print(f"{'Backend':<35} {'Mode':<24} {'B':<5} {'T/H':<6} {'Params':<10} {'ms/call':<14} {'calls/s':<12}")
    print(f"{'-' * 110}")
    for r in results:
        t_val = r.get("T", r.get("H", ""))
        ms = r.get("ms_per_call", float("nan"))
        cps = r.get("calls_per_sec", float("nan"))
        err = r.get("error", "")
        print(f"{r['label']:<35} {r['mode']:<24} {r['B']:<5} {str(t_val):<6} {r['params']:<10} {ms:<14.4f} {cps:<12.1f} {err}")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    main()
