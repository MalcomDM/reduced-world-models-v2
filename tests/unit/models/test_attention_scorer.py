import pytest, torch, math
from torch import Tensor

from rwm.config.config import TOKEN_DIM, QUERY_DIM
from rwm.models.rwm_deterministic.attention_scorer import AttentionScorer


@pytest.mark.models
def test_attention_scorer_output_shape() -> None:
    """ Given random tokens of shape (B, N, D), the scorer should output (B, N). """
    scorer = AttentionScorer()
    scorer.eval()

    batch_size = 3
    N = 10
    D = TOKEN_DIM
    tokens: Tensor = torch.randn(batch_size, N, D)

    with torch.no_grad():
        logits: Tensor = scorer(tokens)

    assert logits.shape == (batch_size, N), f"Expected output shape {(batch_size, N)}, got {logits.shape}"
    

@pytest.mark.models
def test_attention_scorer_constant_tokens() -> None:
    """ If all tokens are identical, the attention scores should be identical per token. """
    scorer = AttentionScorer()
    scorer.eval()

    batch_size = 2
    N = 5
    D = TOKEN_DIM
    tokens: Tensor = torch.full((batch_size, N, D), 0.5)

    with torch.no_grad():
        logits: Tensor = scorer(tokens)

    for i in range(batch_size):
        row = logits[i]
        assert torch.allclose(row, row[0].expand_as(row)), (
            "Attention scores vary across identical tokens"
        )


@pytest.mark.models
def test_attention_scorer_manual_weights_and_scaling() -> None:
    """
    By setting to_k to identity and query to all ones (requires TOKEN_DIM==QUERY_DIM),
    raw score for each token = sum(token_vector), and logits = raw / sqrt(D).
    """
    assert TOKEN_DIM == QUERY_DIM, "This test requires TOKEN_DIM == QUERY_DIM"
    scorer = AttentionScorer()
    scorer.eval()

    # Make to_k an identity mapping
    with torch.no_grad():
        scorer.to_k.weight.data = torch.eye(TOKEN_DIM)
    # Ensure no bias
    assert not scorer.to_k.bias, "to_k.bias should be False"

    # Set query vector to all ones
    with torch.no_grad():
        scorer.query.data.fill_(1.0)

    # B = 1
    N = 4
    D = TOKEN_DIM
    # Token i is a vector of all i’s
    base = torch.arange(N, dtype=torch.float32)
    tokens = base.view(N, 1).expand(N, D).unsqueeze(0)  # shape (1, N, D)

    with torch.no_grad():
        logits: Tensor = scorer(tokens)

    # Expected raw = sum(token_i) = i * D ; expected logits = raw / sqrt(D)
    expected = (base * D) / math.sqrt(D)
    scores = logits.squeeze(0)
    for i in range(N):
        assert torch.isclose(scores[i], expected[i], atol=1e-5), (
            f"Token {i}: expected {expected[i]}, got {scores[i]}"
        )
