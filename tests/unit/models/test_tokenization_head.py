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
def test_tokenization_head_is_deterministic_in_mean_mode() -> None:
    """mean eval mode: same input produces identical tokens."""
    head = TokenizationHead(eval_mode="mean").eval()
    x = torch.rand(2, TKN_IN_CHANNELS, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)

    with torch.no_grad():
        first, _, _ = head(x)
        second, _, _ = head(x)

    assert torch.equal(first, second)


@pytest.mark.models
def test_sample_eval_can_vary() -> None:
    """sample eval mode: same input can produce different tokens."""
    head = TokenizationHead(eval_mode="sample").eval()
    x = torch.rand(2, TKN_IN_CHANNELS, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)

    # Run twice with same seed to confirm same seed -> same result
    torch.manual_seed(0)
    with torch.no_grad():
        a, _, _ = head(x)
    torch.manual_seed(0)
    with torch.no_grad():
        b, _, _ = head(x)
    assert torch.equal(a, b)

    # Different seeds -> likely different
    torch.manual_seed(1)
    with torch.no_grad():
        c, _, _ = head(x)
    assert not torch.equal(a, c)


@pytest.mark.models
def test_training_stochastic_regardless_of_eval_mode() -> None:
    """Training always samples, even when eval_mode='mean'."""
    head = TokenizationHead(eval_mode="mean").train()
    x = torch.rand(2, TKN_IN_CHANNELS, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)

    torch.manual_seed(0)
    with torch.no_grad():
        a, _, _ = head(x)
    torch.manual_seed(0)
    with torch.no_grad():
        b, _, _ = head(x)
    # Training uses sampling; same seed -> same result (reparam trick)
    assert torch.equal(a, b)
    # But tokens differ from posterior mean (should not equal mean-only output)
    head.eval()
    with torch.no_grad():
        mean_out, _, _ = head(x)
    assert not torch.equal(a, mean_out)
