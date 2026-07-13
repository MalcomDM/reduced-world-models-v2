import pytest
import torch
from torch import Tensor
from rwm.models.rwm.tokenization_head import TokenizationHead
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
        tokens, _, _  = head(x)

    expected_n = NUM_PATCHES
    assert tokens.shape == (batch_size, expected_n, TOKEN_DIM), (
        f"Expected output shape {(batch_size, expected_n, TOKEN_DIM)}, "
        f"got {tokens.shape}"
    )


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
def test_tokenization_head_is_deterministic_in_eval_mode() -> None:
    """Evaluation must not inject VAE sampling noise into reward metrics."""
    head = TokenizationHead().eval()
    x = torch.rand(2, TKN_IN_CHANNELS, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)

    with torch.no_grad():
        first, _, _ = head(x)
        second, _, _ = head(x)

    assert torch.equal(first, second)
