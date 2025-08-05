import pytest
import torch
from rwm.models.rwm_deterministic.model import ReducedWorldModel
from rwm.config.config import ACTION_DIM, WRNN_HIDDEN_DIM, PATCHES_PER_SIDE, K

@pytest.mark.models
def test_reduced_world_model_forward_shapes_and_determinism() -> None:
    """
    Smoke-test the end-to-end ReducedWorldModel:
      - correct output shapes
      - deterministic behavior in eval mode with force_keep_input=True
    """
    model = ReducedWorldModel(action_dim=ACTION_DIM, dropout_prob=0.0)
    model.eval()

    B = 2
    # Dummy inputs
    img = torch.randn(B, 3, 64, 64)
    a_prev = torch.randn(B, ACTION_DIM)
    h_prev = torch.zeros(B, WRNN_HIDDEN_DIM)
    c_prev = torch.zeros(B, WRNN_HIDDEN_DIM)

    # Forward twice
    h1, c1, r1, mask1, indices1 = model(img, a_prev, h_prev, c_prev, force_keep_input=True)
    h2, c2, r2, mask2, indices2 = model(img, a_prev, h_prev, c_prev, force_keep_input=True)

    # Check shapes
    assert h1.shape == (B, WRNN_HIDDEN_DIM), f"h_new shape: {h1.shape}"
    assert c1.shape == (B, WRNN_HIDDEN_DIM), f"c_new shape: {c1.shape}"
    assert r1.shape == (B, 1), f"r_pred shape: {r1.shape}"
    assert mask1.shape == (B, PATCHES_PER_SIDE * PATCHES_PER_SIDE), (
        f"mask shape: {mask1.shape}, expected {(B, PATCHES_PER_SIDE * PATCHES_PER_SIDE)}"
    )
    assert indices1.shape == (B, K), f"indices shape: {indices1.shape}"

    # Deterministic checks
    assert torch.allclose(h1, h2, atol=1e-6), "h_new not deterministic in eval mode"
    assert torch.allclose(c1, c2, atol=1e-6), "c_new not deterministic in eval mode"
    assert torch.allclose(r1, r2, atol=1e-6), "r_pred not deterministic in eval mode"
    assert torch.equal(mask1, mask2), "mask not deterministic in eval mode"
    assert torch.equal(indices1, indices2), "indices not deterministic in eval mode"

