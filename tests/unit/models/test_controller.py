import pytest
import torch
from torch import Tensor

from rwm.models.controller.model import Controller
from rwm.config.config import WRNN_HIDDEN_DIM, ACTION_DIM


@pytest.mark.models
def test_controller_output_shape_and_range() -> None:
    """
    Controller should map (B, hidden_dim) → (B, action_dim),
    and outputs must lie within [-1, 1].
    """
    model = Controller(hidden_dim=WRNN_HIDDEN_DIM, action_dim=ACTION_DIM)
    model.eval()

    for B in [1, 4]:
        h: Tensor = torch.randn(B, WRNN_HIDDEN_DIM)
        with torch.no_grad():
            actions: Tensor = model(h)

        # shape check
        assert actions.shape == (B, ACTION_DIM), f"Expected shape {(B, ACTION_DIM)}, got {actions.shape}"
        # range check
        assert torch.all(actions <= 1.0) and torch.all(actions >= -1.0), "Actions out of [-1,1] range"


@pytest.mark.models
def test_controller_determinism_in_eval() -> None:
    """ In eval mode, repeated forward calls must be identical. """
    model = Controller()
    model.eval()

    h = torch.randn(2, WRNN_HIDDEN_DIM)
    with torch.no_grad():
        a1 = model(h)
        a2 = model(h)

    assert torch.allclose(a1, a2, atol=1e-6), "Controller outputs vary in eval mode"


@pytest.mark.models
def test_controller_gradient_flow() -> None:
    """ Ensure that training the controller yields gradients in its parameters. """
    model = Controller()
    model.train()

    B = 3
    h: Tensor = torch.randn(B, WRNN_HIDDEN_DIM)
    actions: Tensor = model(h)
    loss: Tensor = actions.mean()
    loss.backward()	# type: ignore

    # At least one parameter should have a non-null gradient
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads), "No gradients in controller parameters"
