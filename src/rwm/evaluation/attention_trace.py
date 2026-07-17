"""Attention instrumentation without altering model inference.

Exposes:
  - all-patch scorer logits (before Top-K)
  - hard Top-K indices
  - pooling weights among selected patches
  - grid coordinates / patch geometry
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from rwm.config.config import (
    FEATURE_MAP_SIZE,
    PATCH_SIZE,
    PATCH_STRIDE,
    PATCHES_PER_SIDE,
)


def patch_grid_coords() -> torch.Tensor:
    """Return ``(P, 2)`` tensor of (y, x) grid coordinates for each patch
    center in the feature-map coordinate system.

    P = PATCHES_PER_SIDE ** 2.
    """
    side = PATCHES_PER_SIDE
    ys, xs = torch.meshgrid(
        torch.arange(side, dtype=torch.float32),
        torch.arange(side, dtype=torch.float32),
        indexing="ij",
    )
    # Scale to feature map coordinates (center of each patch).
    offset = PATCH_SIZE // 2
    stride = PATCH_STRIDE
    coords = torch.stack([
        ys * stride + offset,
        xs * stride + offset,
    ], dim=-1).reshape(-1, 2)  # (P, 2)
    return coords


def image_coords_from_patches(feature_map_size: int = FEATURE_MAP_SIZE) -> torch.Tensor:
    """Map patch grid coordinates to image pixel coordinates.

    The feature map is ``feature_map_size x feature_map_size``.  Each patch
    coordinate is scaled by ``image_size / feature_map_size`` where
    ``image_size = 64``.
    """
    patch_coords = patch_grid_coords()
    scale = 64.0 / feature_map_size
    return patch_coords * scale


# ---------------------------------------------------------------------------
# Trace data container
# ---------------------------------------------------------------------------

class AttentionTrace:
    """Captured attention data for one forward pass.

    Fields
    ------
    logits:
        ``(B, N)`` — scorer logits before Top-K (all patches).
    indices:
        ``(B, K)`` — hard Top-K patch indices.
    weights:
        ``(B, K)`` — softmax attention weights on the K selected patches.
    grid_coords:
        ``(N, 2)`` — (y, x) patch centres in feature-map coordinates.
    image_coords:
        ``(N, 2)`` — (y, x) patch centres in image pixel coordinates.
    selection_mode:
        Active selection mode (``"learned"``, ``"fixed_uniform"``, or
        ``"fixed_random"``).
    selection_k:
        Number of selected tokens (``K``).
    """
    def __init__(
        self,
        logits: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        selection_mode: str = "learned",
        selection_k: int = 8,
    ):
        self.logits = logits.detach().cpu()
        self.indices = indices.detach().cpu()
        self.weights = weights.detach().cpu()
        self.grid_coords = patch_grid_coords()
        self.image_coords = image_coords_from_patches()
        self.selection_mode = selection_mode
        self.selection_k = selection_k


def trace_attention(model, img: torch.Tensor) -> AttentionTrace:
    """Run a single frame through the perception pipeline and capture
    attention data without altering model state or gradients.

    Uses the actual ``spatial_hd.forward()`` to obtain pooling weights,
    ensuring the trace is faithful to the real model path.

    The model is switched to eval mode for the trace and restored after.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        feat = model.encoder(img)
        tok_out = model.tokenizer(feat)
        tokens = tok_out if isinstance(tok_out, torch.Tensor) else tok_out[0]

        logits = model.scorer(tokens)  # (B, N)
        selection_mask, indices = model.selector(logits)  # (B, N), (B, K)

        # Use the real spatial head to get correct pooling weights.
        _h_spatial, attn_k = model.spatial_hd(tokens, logits, selection_mask, indices)

    model.train(was_training)
    sel_mode = getattr(model.selector, "selection_mode", "learned")
    sel_k = getattr(model.selector, "k", 8)
    return AttentionTrace(
        logits=logits, indices=indices, weights=attn_k,
        selection_mode=sel_mode, selection_k=sel_k,
    )


# ---------------------------------------------------------------------------
# Rendering helpers (headless)
# ---------------------------------------------------------------------------

def render_heatmap(
    trace: AttentionTrace,
    image_size: int = 64,
    grid_size: int = FEATURE_MAP_SIZE,
) -> torch.Tensor:
    """Render a softmax-normalised patch-score heatmap as a ``(1, H, W)``
    float tensor (0-1 range) aligned to the input image.

    Each patch's softmax weight is painted into its receptive field area.
    Overlapping receptive fields use their mean score, rather than letting the
    final patch visited overwrite the earlier ones.
    """
    import torch
    B, N = trace.logits.shape
    H = W = image_size

    heatmap = torch.zeros((B, H, W), dtype=torch.float32)
    coverage = torch.zeros((B, H, W), dtype=torch.float32)
    soft = torch.softmax(trace.logits, dim=-1)  # (B, N)

    scale = image_size / grid_size
    patch_cells = int(PATCH_SIZE * scale)
    stride_cells = int(PATCH_STRIDE * scale)

    for b in range(B):
        for n in range(N):
            gy = (n // PATCHES_PER_SIDE) * stride_cells
            gx = (n % PATCHES_PER_SIDE) * stride_cells
            val = float(soft[b, n])
            gy_end = min(gy + patch_cells, H)
            gx_end = min(gx + patch_cells, W)
            heatmap[b, gy:gy_end, gx:gx_end] += val
            coverage[b, gy:gy_end, gx:gx_end] += 1

    heatmap /= coverage.clamp_min(1)

    # Normalise to [0, 1] per image.
    for b in range(B):
        m = heatmap[b].max()
        if m > 0:
            heatmap[b] /= m

    return heatmap


def render_selected_overlay(
    trace: AttentionTrace,
    image_size: int = 64,
    grid_size: int = FEATURE_MAP_SIZE,
) -> torch.Tensor:
    """Render a binary mask ``(1, H, W)`` with 1 at the positions of
    selected (Top-K) patches and 0 elsewhere.

    Useful for overlaying on the RGB frame.
    """
    B, K = trace.indices.shape
    H = W = image_size

    overlay = torch.zeros((B, H, W), dtype=torch.float32)

    scale = image_size / grid_size
    patch_cells = int(PATCH_SIZE * scale)
    stride_cells = int(PATCH_STRIDE * scale)

    for b in range(B):
        for idx in trace.indices[b]:
            n = int(idx)
            gy = (n // PATCHES_PER_SIDE) * stride_cells
            gx = (n % PATCHES_PER_SIDE) * stride_cells
            gy_end = min(gy + patch_cells, H)
            gx_end = min(gx + patch_cells, W)
            overlay[b, gy:gy_end, gx:gx_end] = 1.0

    return overlay
