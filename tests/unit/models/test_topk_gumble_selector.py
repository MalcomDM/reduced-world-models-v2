import pytest
import torch, math
from typing import Set, Tuple

from rwm.models.rwm_deterministic.topk_gumbel_selector import TopKGumbelSelector


@pytest.mark.models
def test_selector_output_shape_and_sum() -> None:
    selector = TopKGumbelSelector(k=3, temp=0.5)
    selector.eval()

    batch_size = 4
    N = 10
    logits = torch.randn(batch_size, N)

    mask, indices = selector(logits)
    # Shape checks
    assert mask.shape == (batch_size, N)
    assert indices.shape == (batch_size, 3)

    # Mask sums to ~1.0 (within numerical tolerance)
    sums = mask.sum(dim=1)
    expected = torch.full_like(sums, selector.k, dtype=sums.dtype)
    assert torch.allclose(sums, expected, atol=1e-6), f"Each mask should sum to k={selector.k}"


@pytest.mark.models
def test_selector_indices_match_mask() -> None:
    selector = TopKGumbelSelector(k=2, temp=1.0)
    selector.eval()

    logits = torch.tensor([[0.1, 2.0, 1.0, 0.5]])
    mask, indices = selector(logits)

    B, N = mask.shape
    k = selector.k

    # Each index in `indices[b]` must correspond to a 1.0 in mask[b]
    for b in range(B):
        idx_list = [int(i) for i in indices[b]]
        assert len(idx_list) == k and len(set(idx_list)) == k

        for j in range(N):
            val = mask[b, j].item()  # Python float
            if j in idx_list:
                # exact hard 1.0 check with float
                assert math.isclose(val, 1.0, abs_tol=1e-6), f"mask[{b},{j}]={val} not ≈1"
            else:
                # soft mask entry must be strictly < 1
                assert val < 1.0, f"mask[{b},{j}]={val} should be <1"


@pytest.mark.models
def test_mask_gradient_flow() -> None:
    # Ensure gradients flow through mask_soft part
    selector = TopKGumbelSelector(k=1, temp=1.0)
    logits = torch.randn(2, 5, requires_grad=True)

    mask, _indices = selector(logits)
    loss = (mask * logits).sum()
    loss.backward()

    # gradient should exist for logits (via mask_soft path)
    assert logits.grad is not None
    assert torch.any(logits.grad != 0)



@pytest.mark.models
def test_selector_training_mode_is_noisy() -> None:
    """
    In training mode, with identical logits, noise should cause variation
    in the hard top-K indices over multiple draws.
    """
    selector = TopKGumbelSelector(k=2, temp=1.0)
    selector.train()

    # Use perfectly uniform logits: only noise will break ties
    logits = torch.ones(1, 4)

    seen: Set[Tuple[int, ...]] = set()
    for _ in range(20):
        _, indices = selector(logits)
        # record the tuple of selected positions
        tup: Tuple[int, ...] = tuple(sorted(int(x) for x in indices[0]))
        seen.add(tup)
        if len(seen) > 1:
            break

    assert len(seen) > 1, "Expected multiple distinct index-sets under noise"