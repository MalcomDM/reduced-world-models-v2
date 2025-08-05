import pytest
import torch
from rwm.models.rwm_deterministic.patch_rnn import PatchRNN
from rwm.config.config import K, TOKEN_DIM, PRNN_HIDDEN_DIM

@pytest.mark.models
def test_patch_rnn_output_shape() -> None:
    """ If all tokens are zeros, the final hidden state should remain zeros. """
    rnn = PatchRNN()
    rnn.eval()

    B, N, D = 3, 10, TOKEN_DIM
    tokens = torch.zeros(B, N, D)
    indices = torch.stack([torch.arange(K) for _ in range(B)], dim=0)		# choose first K indices arbitrarily

    h = rnn(tokens, indices)
    assert h.shape == (B, PRNN_HIDDEN_DIM)


@pytest.mark.models
def test_patch_rnn_deterministic_same_input() -> None:
    """ Running the same input twice (in eval) gives identical outputs. """
    rnn = PatchRNN()
    rnn.eval()

    B, N, D = 2, 5, TOKEN_DIM
    tokens = torch.randn(B, N, D)
    indices = torch.randint(0, N, (B, K))

    h1 = rnn(tokens, indices)
    h2 = rnn(tokens, indices)
    assert torch.allclose(h1, h2), "Outputs vary between runs in eval mode"


@pytest.mark.models
def test_patch_rnn_gradient_flow() -> None:
    """ Ensure gradients flow back through the GRUCell over the selected tokens. """
    rnn = PatchRNN()
    rnn.train()

    B, N, D = 1, 4, TOKEN_DIM
    tokens = torch.randn(B, N, D, requires_grad=True)
    indices = torch.tensor([[0, 2] + [0]*(K-2)])  # repeat indices to fill K

    h = rnn(tokens, indices)
    # simple sum-of-outputs loss
    loss = h.sum()
    loss.backward()

    # Check that gradients reached the token inputs for the selected positions
    # Positions 0 and 2 should have non-zero gradient contributions
    assert tokens.grad is not None, "Expected gradients on input tokens"
    grad = tokens.grad.squeeze(0)  # shape (N, D)
    assert torch.any(grad[0] != 0), "No gradient for token 0"
    assert torch.any(grad[2] != 0), "No gradient for token 2"
    # A non-selected token should have zero gradient
    assert torch.allclose(grad[1], torch.zeros_like(grad[1])), "Unexpected grad for non-selected token"
