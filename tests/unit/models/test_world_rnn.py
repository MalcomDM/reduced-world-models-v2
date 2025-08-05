import pytest
import torch
from rwm.models.rwm_deterministic.world_rnn import WorldRNN
from rwm.config.config import WRNN_HIDDEN_DIM, PRNN_HIDDEN_DIM, OBSERVATIONAL_DROPOUT, ACTION_DIM


@pytest.mark.models
def test_world_rnn_output_shapes_and_determinism() -> None:
    model = WorldRNN(action_dim=ACTION_DIM, dropout_prob=OBSERVATIONAL_DROPOUT)
    model.eval()

    B = 3
    h_prev = torch.zeros(B, WRNN_HIDDEN_DIM)
    c_prev = torch.zeros(B, WRNN_HIDDEN_DIM)
    x_spatial = torch.randn(B, PRNN_HIDDEN_DIM)
    a_prev = torch.randn(B, ACTION_DIM)

    # Test without dropout for determinism
    h1, c1, r1 = model(h_prev, c_prev, x_spatial, a_prev, force_keep_input=True)
    h2, c2, r2 = model(h_prev, c_prev, x_spatial, a_prev, force_keep_input=True)

    assert h1.shape == (B, WRNN_HIDDEN_DIM)
    assert c1.shape == (B, WRNN_HIDDEN_DIM)
    assert r1.shape == (B, 1)
    assert torch.allclose(h1, h2) and torch.allclose(c1, c2) and torch.allclose(r1, r2)


@pytest.mark.models
def test_world_rnn_dropout_behavior() -> None:
    """
    With dropout_prob=1.0 (always drop), 
    x_spatial is masked to zero. 
    If we also zero a_prev, then the cell input is all zeros,
    so forward(force_keep_input=False) should match 
    forward(force_keep_input=True), since both see zero input.
    """
    model = WorldRNN(action_dim=ACTION_DIM, dropout_prob=1.0)
    model.train()

    B = 4
    # random previous hidden/cell
    h_prev = torch.randn(B, WRNN_HIDDEN_DIM)
    c_prev = torch.randn(B, WRNN_HIDDEN_DIM)
    # spatial and action both zero
    zero_spatial = torch.zeros(B, PRNN_HIDDEN_DIM)
    zero_action  = torch.zeros(B, ACTION_DIM)

    # Path 1: dropout applied (x_in = 0)
    h_drop, c_drop, _ = model(
        h_prev, c_prev, zero_spatial, zero_action, force_keep_input=False
    )
    # Path 2: force-keep input (x_in = zero_spatial = 0)
    h_keep, c_keep, _ = model(
        h_prev, c_prev, zero_spatial, zero_action, force_keep_input=True
    )

    # They must be identical
    assert torch.allclose(h_drop, h_keep, atol=1e-6), "Dropout path did not zero spatial input as expected"
    assert torch.allclose(c_drop, c_keep, atol=1e-6), "Dropout path cell-state mismatch"


@pytest.mark.models
def test_world_rnn_gradient_flow() -> None:
    model = WorldRNN(action_dim=ACTION_DIM, dropout_prob=0.0)
    model.train()

    B = 2
    h_prev = torch.randn(B, WRNN_HIDDEN_DIM, requires_grad=True)
    c_prev = torch.randn(B, WRNN_HIDDEN_DIM, requires_grad=True)
    x_spatial = torch.randn(B, PRNN_HIDDEN_DIM, requires_grad=True)
    a_prev = torch.randn(B, ACTION_DIM, requires_grad=True)

    # Forward pass
    h_new, c_new, r_pred = model(h_prev, c_prev, x_spatial, a_prev)
    # Assert output shapes
    assert h_new.shape == (B, WRNN_HIDDEN_DIM)
    assert c_new.shape == (B, WRNN_HIDDEN_DIM)
    assert r_pred.shape == (B, 1)

    # Backward on reward prediction only
    loss = r_pred.sum()
    loss.backward()

    # Ensure gradients have flowed back to all inputs
    assert h_prev.grad is not None, "No gradient for h_prev"
    assert c_prev.grad is not None, "No gradient for c_prev"
    assert x_spatial.grad is not None, "No gradient for x_spatial"
    assert a_prev.grad is not None, "No gradient for a_prev"


@pytest.mark.models
def test_observational_dropout_full_zero():
    model = WorldRNN(action_dim=ACTION_DIM, dropout_prob=1.0)
    model.train()
    B  = 2
    h0 = torch.randn(B, WRNN_HIDDEN_DIM)
    c0 = torch.randn(B, WRNN_HIDDEN_DIM)
    x  = torch.randn(B, PRNN_HIDDEN_DIM)
    a  = torch.randn(B, ACTION_DIM)

    # Force dropout (always drop)
    h_drop, c_drop, _ = model(h0, c0, x, a, force_keep_input=False)
    # Force-keep same inputs
    h_keep, c_keep, _ = model(h0, c0, torch.zeros_like(x), a, force_keep_input=False)
    # They should match
    assert torch.allclose(h_drop, h_keep)
    assert torch.allclose(c_drop, c_keep)