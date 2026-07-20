#!/usr/bin/env python3
"""Stage 6.0 — Joint-Gradient Measurement Gate.

Audits gradient flow from every candidate loss to every parameter block
without modifying optimizer state or performing parameter updates.

Usage::

    # Eval-parity mode (deterministic, tokenizer mean)
    python scripts/diagnostics/audit_joint_gradients.py \\
        --anchor runs/.../checkpoint_best.pt \\
        --ac runs/.../ac_checkpoint_2000.pt \\
        --out /tmp/audit_eval_parity.json

    # Gradient-audit mode (seeded train, tokenizer sample)
    python scripts/diagnostics/audit_joint_gradients.py \\
        --anchor runs/.../checkpoint_best.pt \\
        --ac runs/.../ac_checkpoint_2000.pt \\
        --out /tmp/audit_grad.json \\
        --train-mode --seed 42

    # Smoke test (2 batches, no hash verification overhead)
    python scripts/diagnostics/audit_joint_gradients.py \\
        --anchor runs/.../checkpoint_best.pt \\
        --ac runs/.../ac_checkpoint_2000.pt \\
        --out /tmp/audit_smoke.json \\
        --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rwm.config.config import ACTION_DIM
from rwm.config.experiment_config import ExperimentConfig, TemporalConfig
from rwm.data.rollout_dataset import RolloutDataset
from rwm.evaluation.joint_gradient_audit import (
    run_joint_gradient_audit,
    save_audit_result,
)
from rwm.models.actor_critic import ActorCritic
from rwm.models.rwm.model import ReducedWorldModel
from rwm.trainers.imagined_actor_critic import ImaginedACTrainer, ImaginedACTrainingConfig
from rwm.utils.checkpointing import load_checkpoint, model_from_checkpoint


def _batch_fingerprint(batch: Dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def _load_model_and_ac(
    anchor_path: Path,
    ac_checkpoint_path: Path,
    device: torch.device,
) -> tuple:
    """Load frozen world model and trained Actor-Critic."""
    ckpt = load_checkpoint(anchor_path)
    model = model_from_checkpoint(ckpt, action_dim=ACTION_DIM,
                                   tokenizer_eval_mode_override="mean")
    model.eval()
    model.to(device)

    # Build AC via dummy trainer for loading
    data_root = Path(
        ckpt["config"].data.dataset_dir if ckpt["config"]
        else "data/rollouts/rwm_deterministic/scenario_0"
    )
    dummy_ds = RolloutDataset(
        root_dir=data_root, sequence_len=16, image_size=64,
    )
    dummy_loader = DataLoader(dummy_ds, batch_size=1, shuffle=False, drop_last=True)
    # For SRU, the trainer handles backend detection
    trainer = ImaginedACTrainer(
        model=model, train_loader=dummy_loader,
        train_cfg=ImaginedACTrainingConfig(max_batches=1),
        device=device, out_dir=Path("/tmp/_audit_loader"),
    )
    trainer.load_actor_critic_checkpoint(ac_checkpoint_path)
    return model, trainer.ac


def _load_batch(
    anchor_path: Path,
    device: torch.device,
    smoke: bool = False,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> Dict:
    """Load a single batch from the SRU dataset."""
    ckpt = load_checkpoint(anchor_path)
    exp_config: ExperimentConfig = ckpt["config"]
    data_dir = exp_config.data.dataset_dir
    seq_len = exp_config.data.sequence_len

    burn_in = 20  # SRU burn-in
    cache_path = Path(cache_dir) if cache_dir else None

    ds = RolloutDataset(
        root_dir=Path(data_dir),
        sequence_len=seq_len,
        image_size=exp_config.data.image_size,
        cache_dir=cache_path,
        recurrent_context=True,
        burn_in_steps=burn_in,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        ds, batch_size=2 if smoke else 8, shuffle=True, drop_last=True,
        generator=generator,
    )
    batch = next(iter(loader))
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 6.0 — Joint-Gradient Measurement Gate",
    )
    parser.add_argument(
        "--anchor", type=str, required=True,
        help="Path to frozen world-model anchor (.pt)",
    )
    parser.add_argument(
        "--ac", type=str, required=True,
        help="Path to Actor-Critic checkpoint (.pt)",
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--train-mode", action="store_true",
        help="Enable train-mode for tokenizer sampling / Top-K STE",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for gradient-audit mode",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Reduced batch size and skip hash verification",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="data/cache/rollout_frames_v1",
        help="Frame cache directory",
    )
    parser.add_argument(
        "--horizon", type=int, default=4,
        help="Imagination horizon for AC losses",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", file=sys.stderr)

    anchor_path = Path(args.anchor)
    ac_path = Path(args.ac)

    print(f"Loading anchor: {anchor_path}", file=sys.stderr)
    print(f"Loading AC checkpoint: {ac_path}", file=sys.stderr)

    t0 = time.time()
    model, ac = _load_model_and_ac(anchor_path, ac_path, device)
    load_time = time.time() - t0
    backend = getattr(model, "_temporal_backend", "unknown")
    print(f"Backend: {backend}", file=sys.stderr)
    print(f"Load time: {load_time:.2f}s", file=sys.stderr)

    print(f"Loading batch...", file=sys.stderr)
    cache_dir = args.cache_dir if args.cache_dir else None
    batch = _load_batch(
        anchor_path, device, smoke=args.smoke, cache_dir=cache_dir,
        seed=args.seed,
    )
    print(f"Batch: obs={batch['obs'].shape}, actions={batch['action'].shape}", file=sys.stderr)

    if args.smoke:
        # Trim to 1 sample for speed
        batch = {k: v[:1] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Determine eval-parity mode (default) or gradient-audit mode
    train_mode = args.train_mode
    seed = args.seed if train_mode else None

    print(f"Running audit (train_mode={train_mode}, seed={seed})...", file=sys.stderr)
    t1 = time.time()
    result = run_joint_gradient_audit(
        model=model,
        ac=ac,
        batch=batch,
        horizon=args.horizon,
        entropy_coef=0.03,
        gamma=0.997,
        lambda_=0.95,
        seed=seed or 42,
        train_mode_params=train_mode,
    )
    audit_time = time.time() - t1
    result["runtime_s"] = round(audit_time, 2)
    result["load_time_s"] = round(load_time, 2)
    result["backend"] = backend
    result["smoke"] = args.smoke
    result["train_mode"] = train_mode
    result["batch_fingerprint"] = _batch_fingerprint(batch)

    print(f"Audit time: {audit_time:.2f}s", file=sys.stderr)

    if torch.cuda.is_available():
        result["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 3)

    save_audit_result(result, out_path)
    print(f"Results saved to {out_path}", file=sys.stderr)

    # Print routing summary
    print("\n=== Gradient Routing Summary ===", file=sys.stderr)
    losses = result.get("losses", {})
    for lname, ldata in losses.items():
        if not isinstance(ldata, dict) or "blocks" not in ldata:
            continue
        routes = [bname for bname, bdata in ldata["blocks"].items()
                  if isinstance(bdata, dict) and bdata.get("nonzero_gradient")]
        print(f"  {lname}: reaches {len(routes)} blocks: {routes}", file=sys.stderr)

    # Hash identity check
    if not args.smoke:
        if result.get("hash_identity"):
            print("Hash identity: PASS", file=sys.stderr)
        else:
            print("ERROR: Hash identity FAILED — parameters changed!", file=sys.stderr)
            sys.exit(1)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
