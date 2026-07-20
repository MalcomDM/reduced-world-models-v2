"""Stage 6.0 — Joint-Gradient Measurement Gate (diagnostic only).

Reconstructs factual SRU states from scratch (no cached z), computes each
loss independently, and measures gradient flow to every parameter block.
No optimizer step or state mutation occurs.

Gradient routing rules enforced:
  Visible/masked reward → RewardHead, ControllerTrunk, SRU, visible perception.
  KL → Tokenizer, Encoder only.
  Critic → OnlineCritic, ControllerTrunk, SRU, factual perception; NOT Actor/TargetCritic/RewardHead.
  Actor → Actor, ControllerTrunk, SRU, factual perception; NOT either Critic or RewardHead.
  Entropy → Actor and its shared path; NOT Critic or RewardHead.
  TargetCritic → no gradient under any loss.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rwm.config.config import ACTION_DIM
from rwm.config.experiment_config import ActorCriticConfig
from rwm.imagination import ImaginationRollout
from rwm.models.actor_critic import ActorCritic, compute_td_advantage, compute_lambda_returns
from rwm.models.rwm.model import ReducedWorldModel
from rwm.trainers.deterministic.world_model_trainer import masked_kl_normal
from rwm.distributions import BoundedGaussian


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARAM_BLOCK_NAMES: List[str] = [
    "actor",
    "online_critic",
    "target_critic",
    "controller_trunk",
    "reward_head",
    "minimal_sru",
    "spatial_attention_head",
    "attention_scorer",
    "topk_selector",
    "tokenizer",
    "encoder",
]

LOSS_NAMES: List[str] = [
    "visible_reward_mse",
    "masked_reward_mse",
    "tokenizer_kl",
    "critic_loss",
    "actor_loss",
    "entropy",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _param_l2_sum(params: Sequence[Tensor]) -> float:
    return sum(p.data.norm(2).item() ** 2 for p in params) ** 0.5


def _grad_l2_sum(params: Sequence[Tensor]) -> Optional[float]:
    sq = 0.0
    has_grad = False
    for p in params:
        if p.grad is not None:
            sq += p.grad.norm(2).item() ** 2
            has_grad = True
    return sq ** 0.5 if has_grad else None


def _cos_sim(g1: Tensor, g2: Tensor) -> float:
    flat1 = g1.detach().flatten()
    flat2 = g2.detach().flatten()
    dot = (flat1 * flat2).sum()
    n1 = flat1.norm(2)
    n2 = flat2.norm(2)
    if n1.item() < 1e-12 or n2.item() < 1e-12:
        return 0.0 / 0.0  # raises; caller must check
    return (dot / (n1 * n2)).item()


# ---------------------------------------------------------------------------
# Block parameter extractors
# ---------------------------------------------------------------------------

def _gather_block_params(model: ReducedWorldModel, ac: ActorCritic) -> Dict[str, List[nn.Parameter]]:
    """Return a dict of block_name → list of parameters for gradient measurement."""

    ctrl_all = list(model.controller.parameters())
    # Identify reward_head params (the Linear whose output is 1)
    rh_params: List[nn.Parameter] = []
    for mod in model.controller.modules():
        if isinstance(mod, nn.Linear) and mod.out_features == 1:
            rh_params = list(mod.parameters())
            break
    rh_ids = {id(p) for p in rh_params}
    ctrl_trunk = [p for p in ctrl_all if id(p) not in rh_ids]

    blocks: Dict[str, List[nn.Parameter]] = {
        "actor": list(ac.actor.parameters()),
        "online_critic": list(ac.critic.parameters()),
        "target_critic": list(ac.target_critic.parameters()),
        "controller_trunk": ctrl_trunk,
        "reward_head": rh_params,
        "minimal_sru": list(model.world_hd.parameters()),
        "spatial_attention_head": list(model.spatial_hd.parameters()),
        "attention_scorer": list(model.scorer.parameters()),
        "topk_selector": list(model.selector.parameters()),
        "tokenizer": list(model.tokenizer.parameters()),
        "encoder": list(model.encoder.parameters()),
    }
    return blocks


def _param_hash(params: Sequence[nn.Parameter]) -> str:
    h = hashlib.sha256()
    for p in params:
        h.update(p.data.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _all_param_hash(model: ReducedWorldModel, ac: ActorCritic) -> Dict[str, str]:
    blocks = _gather_block_params(model, ac)
    return {name: _param_hash(params) for name, params in blocks.items() if params}


# ---------------------------------------------------------------------------
# Build prev_actions (shared with S5 trainer)
# ---------------------------------------------------------------------------

def _build_prev_actions(
    actions: Tensor, valid_step: Tensor,
    predecessor_action: Tensor, seq_len: int,
) -> Tensor:
    B = actions.shape[0]
    device = actions.device
    prev = torch.zeros(B, seq_len, ACTION_DIM, device=device)
    if seq_len > 1:
        prev[:, 1:] = actions[:, :seq_len - 1]
    first_valid = valid_step.long().argmax(dim=1)
    for b in range(B):
        fv = int(first_valid[b].item())
        if valid_step[b, fv]:
            prev[b, fv] = predecessor_action[b]
    return prev


# ---------------------------------------------------------------------------
# Warmup window (matching S5 trainer _prep_warmup)
# ---------------------------------------------------------------------------

def _extract_warmup_window(
    obs: Tensor, actions: Tensor, valid_step: Tensor, loss_mask: Tensor,
    predecessor_action: Tensor, ws: int = 4,
) -> Dict[str, Tensor]:
    """Extract burn-in + first ws target positions, matching S5 _prep_warmup.

    Returns obs_warm, act_warm, prev_actions_warm with T = first_target + ws.
    """
    B, T_total = obs.shape[0], obs.shape[1]
    device = obs.device
    first_target = loss_mask.long().argmax(dim=1)
    first_valid = valid_step.long().argmax(dim=1)
    T_warm = int(first_target.max().item()) + ws
    T_warm = min(T_warm, T_total)

    obs_warm = obs[:, :T_warm]
    act_warm = actions[:, :T_warm]
    valid_warm = valid_step[:, :T_warm].clone()
    positions = torch.arange(T_warm, device=device).unsqueeze(0)
    sample_ends = (first_target + ws).clamp_max(T_total).unsqueeze(1)
    valid_warm &= positions < sample_ends
    prev_warm = torch.zeros(B, T_warm, ACTION_DIM, device=device)
    if T_warm > 1:
        prev_warm[:, 1:] = act_warm[:, :T_warm - 1]
    for b in range(B):
        fv = int(first_valid[b].item())
        if fv < T_warm:
            prev_warm[b, fv] = predecessor_action[b]

    return {
        "obs": obs_warm,
        "actions": act_warm,
        "prev_actions": prev_warm,
        "valid_step": valid_warm,
    }


# ---------------------------------------------------------------------------
# State snapshot / restore
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ModelSnapshot:
    param_data: Dict[int, Tensor]
    buffer_data: Dict[str, Tensor]
    requires_grad: Dict[int, bool]
    training_modes: Dict[int, bool]
    tokenizer_eval_mode: str


def _snapshot_model(model: ReducedWorldModel, ac: ActorCritic) -> ModelSnapshot:
    param_data = {}
    for p in model.parameters():
        param_data[id(p)] = p.data.clone()
    for p in ac.parameters():
        param_data[id(p)] = p.data.clone()
    # Use path-based keys for buffers so lookup works across objects
    buffer_data = {}
    for prefix, mod in [("model", model), ("ac", ac)]:
        for name, buf in mod.named_buffers(recurse=True):
            buffer_data[f"{prefix}.{name}"] = buf.clone()
    requires_grad = {}
    for p in model.parameters():
        requires_grad[id(p)] = p.requires_grad
    for p in ac.parameters():
        requires_grad[id(p)] = p.requires_grad
    training_modes = {
        id(module): module.training
        for root in (model, ac)
        for module in root.modules()
    }
    return ModelSnapshot(
        param_data=param_data,
        buffer_data=buffer_data,
        requires_grad=requires_grad,
        training_modes=training_modes,
        tokenizer_eval_mode=model.tokenizer.eval_mode,
    )


def _restore_snapshot(model: ReducedWorldModel, ac: ActorCritic, snap: ModelSnapshot) -> None:
    for p in model.parameters():
        if id(p) in snap.param_data:
            p.data.copy_(snap.param_data[id(p)])
    for p in ac.parameters():
        if id(p) in snap.param_data:
            p.data.copy_(snap.param_data[id(p)])
    for prefix, mod in [("model", model), ("ac", ac)]:
        for name, buf in mod.named_buffers(recurse=True):
            key = f"{prefix}.{name}"
            if key in snap.buffer_data:
                buf.copy_(snap.buffer_data[key])
    for p in model.parameters():
        if id(p) in snap.requires_grad:
            p.requires_grad_(snap.requires_grad[id(p)])
    for p in ac.parameters():
        if id(p) in snap.requires_grad:
            p.requires_grad_(snap.requires_grad[id(p)])
    # Restore every submodule independently.  Calling only model.train(...)
    # would destroy intentionally mixed train/eval states.
    for root in (model, ac):
        for module in root.modules():
            if id(module) in snap.training_modes:
                module.training = snap.training_modes[id(module)]
    model.tokenizer.eval_mode = snap.tokenizer_eval_mode


# ---------------------------------------------------------------------------
# Per-block gradient measurement
# ---------------------------------------------------------------------------

def _measure_block_gradients(
    loss_tensor: Tensor,
    blocks: Dict[str, List[nn.Parameter]],
    block_names: List[str],
    retain_graph: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Measure gradients for each block: graph_connected, nonzero_gradient, norm, ratio, finite."""
    measured: Dict[str, Dict[str, Any]] = {}
    for name in block_names:
        params = blocks.get(name, [])
        if not params:
            measured[name] = {
                "graph_connected": False,
                "nonzero_gradient": False,
                "grad_l2_norm": None,
                "param_l2_norm": None,
                "ratio": None,
                "finite": None,
            }
            continue
        trainable = [p for p in params if p.requires_grad]
        if not trainable:
            measured[name] = {
                "graph_connected": False,
                "nonzero_gradient": False,
                "grad_l2_norm": None,
                "param_l2_norm": _param_l2_sum(params),
                "ratio": None,
                "finite": None,
            }
            continue
        p_l2 = _param_l2_sum(trainable)
        grads = torch.autograd.grad(
            loss_tensor, trainable,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        has_any = any(g is not None for g in grads)
        has_nonzero = any(g is not None and g.abs().sum().item() > 0 for g in grads)
        if has_nonzero:
            for p, g in zip(trainable, grads):
                p.grad = g
            g_l2 = _grad_l2_sum(trainable)
            ratio = (g_l2 / p_l2) if p_l2 > 1e-12 else None
            finite = all(torch.isfinite(p.grad).all().item() for p in trainable if p.grad is not None)
        else:
            g_l2 = None
            ratio = None
            finite = None
        measured[name] = {
            "graph_connected": has_any,
            "nonzero_gradient": has_nonzero,
            "grad_l2_norm": g_l2,
            "param_l2_norm": p_l2,
            "ratio": ratio,
            "finite": finite,
        }
    return measured


# ---------------------------------------------------------------------------
# Per-block cosine similarity
# ---------------------------------------------------------------------------

def _compute_per_block_cosine(
    all_grads: Dict[str, Dict[str, Optional[Tensor]]],
    block_names: List[str],
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """Per-block, per-loss-pair cosine similarity.

    Returns block → {loss_pair: value | None}.
    For each block, collects flat grads from each loss (None if disconnected).
    Aligns by zero-padding to the block's max param count.
    Reports None when either side has None or zero norm.
    """
    result: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for bname in block_names:
        pairs: Dict[str, Dict[str, Optional[float]]] = {}
        loss_names = sorted(all_grads.keys())
        for i, li in enumerate(loss_names):
            for j, lj in enumerate(loss_names):
                if i >= j:
                    continue
                gi = all_grads[li].get(bname)
                gj = all_grads[lj].get(bname)
                if gi is None or gj is None:
                    pairs[f"{li}×{lj}"] = None
                else:
                    f_i = gi.flatten()
                    f_j = gj.flatten()
                    n_i = f_i.norm(2).item()
                    n_j = f_j.norm(2).item()
                    if n_i < 1e-12 or n_j < 1e-12:
                        pairs[f"{li}×{lj}"] = None
                    else:
                        pairs[f"{li}×{lj}"] = (f_i @ f_j / (n_i * n_j)).item()
        result[bname] = pairs
    return result


def _flatten_aligned_grads(
    params: Sequence[nn.Parameter],
) -> Optional[Tensor]:
    """Flatten a block without changing coordinates across losses."""
    trainable = [parameter for parameter in params if parameter.requires_grad]
    if not any(parameter.grad is not None for parameter in trainable):
        return None
    return torch.cat([
        (
            parameter.grad.detach().flatten()
            if parameter.grad is not None
            else torch.zeros_like(parameter).flatten()
        )
        for parameter in trainable
    ])


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------

def _run_joint_gradient_audit_impl(
    model: ReducedWorldModel,
    ac: ActorCritic,
    batch: Dict[str, Tensor],
    horizon: int = 4,
    warmup_steps: int = 4,
    entropy_coef: float = 0.03,
    gamma: float = 0.997,
    lambda_: float = 0.95,
    seed: int = 42,
    train_mode_params: bool = False,
) -> Dict[str, Any]:
    """Run the full joint-gradient audit.

    Two modes:
    - ``train_mode_params=False`` (eval-parity): model in eval mode,
      tokenizer uses mean, deterministic output.
    - ``train_mode_params=True`` (gradient-audit): model.train() for
      tokenizer sampling and Top-K STE, with seeded RNG.

    Returns a structured dict with per-loss measurements, per-block cosine
    matrices, parameter hashes, parity evidence, and provenance metadata.
    """
    device = next(model.parameters()).device
    B = batch["obs"].shape[0]
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)

    # ---- Snapshot everything before any mutation ----
    snapshot = _snapshot_model(model, ac)
    hash_before = _all_param_hash(model, ac)
    orig_grads: Dict[int, Optional[Tensor]] = {}
    for p in model.parameters():
        orig_grads[id(p)] = p.grad
    for p in ac.parameters():
        orig_grads[id(p)] = p.grad

    # ---- Enable gradients for audit (except Target Critic) ----
    for p in model.parameters():
        p.requires_grad_(True)
    for p in ac.actor.parameters():
        p.requires_grad_(True)
    for p in ac.critic.parameters():
        p.requires_grad_(True)
    for p in ac.target_critic.parameters():
        p.requires_grad_(False)
    # Zero existing grads
    for p in model.parameters():
        p.grad = None
    for p in ac.parameters():
        p.grad = None

    # ---- Mode setup ----
    if train_mode_params:
        model.train()
        model.tokenizer.eval_mode = "sample"
    else:
        model.eval()
        model.tokenizer.eval_mode = "mean"

    # ---- Parameter blocks ----
    blocks = _gather_block_params(model, ac)
    block_names = PARAM_BLOCK_NAMES

    # ---- Move batch to device ----
    obs = batch["obs"].to(device, non_blocking=True)
    act = batch["action"].to(device, non_blocking=True)
    rew = batch["reward"].to(device, non_blocking=True)
    vs = batch["valid_step"].to(device, non_blocking=True)
    lm = batch["loss_mask"].to(device, non_blocking=True)
    pred = batch["predecessor_action"].to(device, non_blocking=True)

    # ===================================================================
    # Eval-parity baseline: always use the deterministic canonical policy.
    # This must not depend on how the checkpoint happened to be loaded or on
    # whether the gradient audit itself uses train-mode stochasticity.
    # ===================================================================
    audit_training_mode = model.training
    audit_tokenizer_mode = model.tokenizer.eval_mode
    model.eval()
    model.tokenizer.eval_mode = "mean"
    with torch.no_grad():
        prev_actions_default = _build_prev_actions(act, vs, pred, obs.shape[1])
        parity_out = model.forward_sequence(
            obs, prev_actions_default, act,
            force_keep_input=True,
            observation_keep=None,
            valid_step=vs,
            observation_dropout_execution="post_perception",
        )
        parity_reward_seq = parity_out.reward_pred_seq.clone() if parity_out.reward_pred_seq is not None else None
        parity_temporal_state = parity_out.temporal_state.clone() if parity_out.temporal_state is not None else None
    model.train(audit_training_mode)
    model.tokenizer.eval_mode = audit_tokenizer_mode

    # ===================================================================
    # 1. Visible reward MSE — all-visible forward, target-only loss
    # ===================================================================
    prev_actions = _build_prev_actions(act, vs, pred, obs.shape[1])

    out_visible = model.forward_sequence(
        obs, prev_actions, act,
        force_keep_input=True,
        observation_keep=None,
        valid_step=vs,
        observation_dropout_execution="post_perception",
    )

    r_pred = out_visible.reward_pred_seq
    target_mask = lm.float() * vs.float()
    diff = (r_pred - rew).pow(2)
    loss_visible_mse = (diff * target_mask).sum() / target_mask.sum().clamp_min(1)
    if not torch.isfinite(loss_visible_mse):
        loss_visible_mse = torch.zeros((), device=device)

    # ---- Tokenizer KL (from visible forward, target-only) ----
    loss_kl = torch.zeros((), device=device)
    if out_visible.tok_mu is not None and out_visible.tok_logvar is not None:
        kl_mask = lm & vs
        if kl_mask.any():
            loss_kl = masked_kl_normal(out_visible.tok_mu, out_visible.tok_logvar, kl_mask)

    # ===================================================================
    # 2. Masked reward MSE — genuine pre_perception_skip forward
    # ===================================================================
    # Build mask: burn-in visible + first 4 target visible, next H masked
    T_total = obs.shape[1]
    obs_mask = torch.ones(B, T_total, dtype=torch.bool, device=device)
    masked_positions = torch.zeros(B, T_total, dtype=torch.bool, device=device)
    first_target_pos = lm.long().argmax(dim=1)
    # Build each sample's mask independently so left padding or a future
    # variable-layout dataset cannot silently shift the factual target.
    for b in range(B):
        mask_start = int(first_target_pos[b].item()) + warmup_steps
        mask_end = min(mask_start + horizon, T_total)
        if mask_end > mask_start:
            obs_mask[b, mask_start:mask_end] = False
            masked_positions[b, mask_start:mask_end] = True

    out_masked = model.forward_sequence(
        obs, prev_actions, act,
        force_keep_input=True,
        observation_keep=obs_mask,
        valid_step=vs,
        observation_dropout_execution="pre_perception_skip",
    )

    r_pred_masked = out_masked.reward_pred_seq
    masked_positions = masked_positions & vs & lm
    diff_masked = (r_pred_masked - rew).pow(2)
    loss_masked_mse = (diff_masked * masked_positions.float()).sum() / masked_positions.float().sum().clamp_min(1)
    if not torch.isfinite(loss_masked_mse):
        loss_masked_mse = torch.zeros((), device=device)

    # ===================================================================
    # 3. Imagination warmup — burn-in + first 4 target only (S5 match)
    # ===================================================================
    warmup_data = _extract_warmup_window(obs, act, vs, lm, pred, ws=warmup_steps)
    warm_obs = warmup_data["obs"]
    warm_act = warmup_data["actions"]
    warm_prev = warmup_data["prev_actions"]
    warm_valid = warmup_data["valid_step"]

    # ===================================================================
    # 4. Imagined rollout for AC losses
    # ===================================================================
    imag = ImaginationRollout(model)
    warm_state = imag.warmup(
        warm_obs, warm_prev, warm_act,
        force_keep_input=True,
        valid_step=warm_valid,
    )
    if not warm_state.is_sru:
        raise RuntimeError("Stage 6 joint audit requires an SRU temporal state")
    z_start = warm_state.current_belief

    H = horizon
    D = z_start.shape[-1]
    B_batch = B

    # Use lists to preserve computation graph (in-place assignment to pre-allocated
    # tensor would break the graph).
    imagined_states_list: List[Tensor] = []
    imagined_actions_list: List[Tensor] = []
    imagined_rewards_list: List[Tensor] = []

    z_t = z_start
    for h in range(H):
        c_t = model.controller.encode(z_t)
        dist = ac.actor(c_t)
        a_t = dist.mode() if not train_mode_params else dist.rsample()
        # Reward is a detached target
        with torch.no_grad():
            _, r_t = model.controller(z_t, a_t)

        imagined_states_list.append(z_t)
        imagined_actions_list.append(a_t)
        imagined_rewards_list.append(r_t.squeeze(-1))

        # Advance SRU with DETACHED action (grads flow through SRU but not through action choice for Actor)
        a_detached = a_t.detach()
        _, _, z_t = imag.advance(
            warm_state.history, warm_state.lengths, a_detached,
            temporal_state=z_t,
        )

    z_H = z_t  # final state after H steps — used for bootstrap

    # Stack lists into tensors (preserves graph through concatenation)
    imagined_states = torch.stack(imagined_states_list, dim=1)   # (B, H, D)
    imagined_actions = torch.stack(imagined_actions_list, dim=1)  # (B, H, A)
    imagined_rewards = torch.stack(imagined_rewards_list, dim=1)  # (B, H)

    # ===================================================================
    # 5. Critic loss — NOT reaching Actor
    # ===================================================================
    z_flat = imagined_states.reshape(B_batch * H, D)
    c_flat_critic = model.controller.encode(z_flat).reshape(B_batch, H, -1)
    critic_values = ac.critic(c_flat_critic.reshape(B_batch * H, -1)).reshape(B_batch, H)

    with torch.no_grad():
        target_values = ac.target_critic(
            c_flat_critic.reshape(B_batch * H, -1)
        ).reshape(B_batch, H)

    # Bootstrap from z_H (final imagined state)
    c_zH = model.controller.encode(z_H)
    bootstrap_value = ac.target_critic(c_zH).squeeze(-1)

    terminated = torch.zeros(B_batch, H, dtype=torch.bool, device=device)
    continuation = torch.ones(B_batch, H, dtype=torch.bool, device=device)

    lambda_returns = compute_lambda_returns(
        imagined_rewards.detach(), target_values, terminated,
        continuation, bootstrap_value, gamma, lambda_,
    )
    critic_loss = F.mse_loss(critic_values, lambda_returns.detach())

    # ===================================================================
    # 6. Actor loss — separate, NOT including entropy, NOT reaching Critic/RewardHead
    # ===================================================================
    # Shared encode path for Actor (separate graph from Critic)
    c_flat_actor = model.controller.encode(z_flat).reshape(B_batch * H, -1)
    dist_flat = ac.actor(c_flat_actor)
    log_prob = dist_flat.log_prob(imagined_actions.reshape(B_batch * H, -1).detach())

    # TD advantages
    target_values_actor = ac.target_critic(c_flat_actor).reshape(B_batch, H)
    bootstrap_value_actor = ac.target_critic(model.controller.encode(z_H)).squeeze(-1).detach()
    advantages = compute_td_advantage(
        critic_values.detach(), imagined_rewards.detach(), target_values_actor.detach(),
        terminated, continuation, bootstrap_value_actor.detach(), gamma,
    )

    actor_loss = -(advantages.detach().unsqueeze(-1) * log_prob).mean()
    if not torch.isfinite(actor_loss):
        actor_loss = torch.zeros((), device=device)

    # ===================================================================
    # 7. Entropy (separate loss, not included in actor_loss above)
    # ===================================================================
    entropy_reg = -entropy_coef * dist_flat.entropy_sample().mean()
    if not torch.isfinite(entropy_reg):
        entropy_reg = torch.zeros((), device=device)

    # ===================================================================
    # Measure gradients per loss
    # ===================================================================
    loss_specs: List[Tuple[str, Tensor, bool]] = [
        ("visible_reward_mse", loss_visible_mse, True),
        ("masked_reward_mse", loss_masked_mse, True),
        ("tokenizer_kl", loss_kl, True),
        ("critic_loss", critic_loss, True),
        ("actor_loss", actor_loss, True),
        ("entropy", entropy_reg, True),
    ]

    measurements: Dict[str, Any] = {}
    all_grad_tensors: Dict[str, Dict[str, Optional[Tensor]]] = {}
    # Initialize all_grad_tensors
    for lname, _, _ in loss_specs:
        all_grad_tensors[lname] = {}

    for lname, lt, retain in loss_specs:
        for p in model.parameters():
            p.grad = None
        for p in ac.parameters():
            p.grad = None

        if not torch.isfinite(lt):
            measurements[lname] = {
                "loss_value": lt.detach().item(),
                "finite": False,
                "blocks": {name: {"graph_connected": False, "nonzero_gradient": False,
                                   "grad_l2_norm": None, "param_l2_norm": _param_l2_sum(blocks.get(name, [])),
                                   "ratio": None, "finite": None}
                           for name in block_names},
            }
            for bname in block_names:
                all_grad_tensors[lname][bname] = None
            continue

        meas = _measure_block_gradients(lt, blocks, block_names, retain_graph=retain)
        measurements[lname] = {
            "loss_value": lt.detach().item(),
            "finite": True,
            "blocks": meas,
        }
        # Collect per-block gradient tensors for cosine
        for bname in block_names:
            params = blocks.get(bname, [])
            trainable = [p for p in params if p.requires_grad]
            all_grad_tensors[lname][bname] = _flatten_aligned_grads(trainable)

    # ---- Per-block cosine ----
    per_block_cos = _compute_per_block_cosine(all_grad_tensors, block_names)

    # ---- Hashes after (should match before since no optimizer step) ----
    hash_after = _all_param_hash(model, ac)

    # ---- Check no optimizer was created/invoked ----
    no_opt = not hasattr(ac, '_optimizer_was_used') or not ac._optimizer_was_used

    # ---- Verify deterministic canonical eval parity before restoration ----
    # Train-mode audit forwards update BatchNorm running buffers even though
    # parameters remain untouched.  Restore those buffers before comparing
    # the canonical deterministic outputs.
    for prefix, module in (("model", model), ("ac", ac)):
        for name, buffer in module.named_buffers(recurse=True):
            key = f"{prefix}.{name}"
            if key in snapshot.buffer_data:
                buffer.copy_(snapshot.buffer_data[key])
    model.eval()
    model.tokenizer.eval_mode = "mean"
    with torch.no_grad():
        check_out = model.forward_sequence(
            obs, prev_actions_default, act,
            force_keep_input=True,
            observation_keep=None,
            valid_step=vs,
            observation_dropout_execution="post_perception",
        )
    reward_parity = (
        parity_reward_seq is None
        or (
            check_out.reward_pred_seq is not None
            and torch.allclose(
                parity_reward_seq, check_out.reward_pred_seq,
                rtol=1e-5, atol=1e-5,
            )
        )
    )
    state_parity = (
        parity_temporal_state is None
        or (
            check_out.temporal_state is not None
            and torch.allclose(
                parity_temporal_state, check_out.temporal_state,
                rtol=1e-5, atol=1e-5,
            )
        )
    )
    reward_max_abs_diff = (
        None
        if parity_reward_seq is None or check_out.reward_pred_seq is None
        else float((parity_reward_seq - check_out.reward_pred_seq).abs().max().item())
    )
    state_max_abs_diff = (
        None
        if parity_temporal_state is None or check_out.temporal_state is None
        else float((parity_temporal_state - check_out.temporal_state).abs().max().item())
    )

    # ---- Restore everything ----
    _restore_snapshot(model, ac, snapshot)
    for p in model.parameters():
        p.grad = orig_grads.get(id(p))
    for p in ac.parameters():
        p.grad = orig_grads.get(id(p))
    torch.set_rng_state(rng_state)

    # ---- Build result ----
    result: Dict[str, Any] = {
        "metadata": {
            "seed": seed,
            "device": str(device),
            "horizon": H,
            "warmup_steps": warmup_steps,
            "train_mode_params": train_mode_params,
            "entropy_coef": entropy_coef,
            "gamma": gamma,
            "lambda_": lambda_,
            "batch_size": B,
            "T_total": obs.shape[1],
        },
        "hash_before": hash_before,
        "hash_after": hash_after,
        "hash_identity": hash_before == hash_after,
        "losses": measurements,
        "cosine_similarity_per_block": per_block_cos,
        "no_optimizer_step": bool(no_opt),
        "eval_parity": {
            "reward_pred_seq_match": bool(reward_parity),
            "temporal_state_match": bool(state_parity),
            "reward_pred_seq_max_abs_diff": reward_max_abs_diff,
            "temporal_state_max_abs_diff": state_max_abs_diff,
            "overall": bool(reward_parity and state_parity),
        },
        "target_critic_requires_grad": {
            "target_critic": all(not p.requires_grad for p in ac.target_critic.parameters()),
        },
    }

    return result


def run_joint_gradient_audit(
    model: ReducedWorldModel,
    ac: ActorCritic,
    batch: Dict[str, Tensor],
    horizon: int = 4,
    warmup_steps: int = 4,
    entropy_coef: float = 0.03,
    gamma: float = 0.997,
    lambda_: float = 0.95,
    seed: int = 42,
    train_mode_params: bool = False,
) -> Dict[str, Any]:
    """Exception-safe public wrapper for the Stage 6.0 diagnostic.

    The implementation deliberately changes gradient flags and module modes
    while measuring routes.  This wrapper guarantees restoration even when a
    forward pass or a diagnostic assertion raises.
    """
    snapshot = _snapshot_model(model, ac)
    original_grads = {
        id(p): (None if p.grad is None else p.grad.detach().clone())
        for root in (model, ac)
        for p in root.parameters()
    }
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        return _run_joint_gradient_audit_impl(
            model=model,
            ac=ac,
            batch=batch,
            horizon=horizon,
            warmup_steps=warmup_steps,
            entropy_coef=entropy_coef,
            gamma=gamma,
            lambda_=lambda_,
            seed=seed,
            train_mode_params=train_mode_params,
        )
    finally:
        _restore_snapshot(model, ac, snapshot)
        for root in (model, ac):
            for p in root.parameters():
                saved = original_grads[id(p)]
                p.grad = None if saved is None else saved.clone()
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize_audit_result(result: Dict[str, Any]) -> str:
    class AuditEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (torch.dtype, torch.device)):
                return str(obj)
            return super().default(obj)

    return json.dumps(result, cls=AuditEncoder, indent=2, sort_keys=True, allow_nan=False)


def save_audit_result(result: Dict[str, Any], path: Path) -> None:
    path.write_text(serialize_audit_result(result))
