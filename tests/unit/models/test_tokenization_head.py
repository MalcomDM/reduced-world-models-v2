import pytest
import torch
from torch import Tensor
from rwm.models.rwm_deterministic.tokenization_head import TokenizationHead
from rwm.config.config import FEATURE_MAP_SIZE, TKN_IN_CHANNELS, TOKEN_DIM, NUM_PATCHES


@pytest.mark.models
def test_tokenization_head_output_shape() -> None:
    """
    Given a feature map of shape (B, C, H, W),
    the TokenizationHead should output (B, N, TOKEN_DIM),
    where N = NUM_PATCH_DIM * NUM_PATCH_DIM.
    """
    head = TokenizationHead()
    head.eval()

    batch_size = 2
    C = TKN_IN_CHANNELS
    H = W = FEATURE_MAP_SIZE
    x: Tensor = torch.rand(batch_size, C, H, W)

    with torch.no_grad():
        tokens: Tensor = head(x)

    expected_n = NUM_PATCHES
    assert tokens.shape == (batch_size, expected_n, TOKEN_DIM), (
        f"Expected output shape {(batch_size, expected_n, TOKEN_DIM)}, "
        f"got {tokens.shape}"
    )


@pytest.mark.models
def test_tokenization_head_deterministic() -> None:
    """ In eval mode, running the same input twice should yield identical outputs. """
    head = TokenizationHead()
    head.eval()

    x = torch.rand(1, TKN_IN_CHANNELS, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
    with torch.no_grad():
        out1 = head(x)
        out2 = head(x)

    assert torch.allclose(out1, out2, atol=0, rtol=0), "Outputs differ between runs in eval mode"


@pytest.mark.models
def test_positional_embedding_buffer() -> None:
    """ The positional embedding buffer should be registered and have the correct shape. """
    head = TokenizationHead()
    assert hasattr(head, "pos_embed_buffer"), "pos_embed_buffer not registered"
    buf: Tensor = head.pos_embed_buffer
    expected_n = NUM_PATCHES
    assert buf.ndim == 3 and buf.shape == (1, expected_n, TOKEN_DIM), (
        f"Expected pos_embed_buffer shape (1, {expected_n}, {TOKEN_DIM}), got {buf.shape}"
    )
    # Ensure buffer is not trainable
    assert not head.pos_embed_buffer.requires_grad, "pos_embed_buffer should not require grad"


@pytest.mark.models
def test_forward_uses_projection_and_unfold() -> None:
    """
    Test that forward applies unfolding and linear projection:
    When the projection weights are zero and bias is zero, the output tokens should equal
    the positional embedding broadcasted.
    """
    head = TokenizationHead()
    head.eval()
    # Zero out projection weights and bias
    head.projection.weight.data.zero_()
    head.projection.bias.data.zero_()

    batch_size = 3
    C = TKN_IN_CHANNELS
    H = W = FEATURE_MAP_SIZE
    x = torch.randn(batch_size, C, H, W)
    with torch.no_grad():
        tokens = head(x)

    # If projection produces zeros, tokens should equal pos_embed_buffer repeated across batch
    buf = head.pos_embed_buffer  # shape (1, N, TOKEN_DIM)
    expected = buf.expand(batch_size, -1, -1)
    assert torch.allclose(tokens, expected), "Tokens do not match positional embeddings when projection is zero"

