import torch
import pytest
import sys
from typing import Optional

@pytest.mark.integration
def test_cuda_available():
    assert torch.cuda.is_available(), "❌ CUDA runtime not available"

    # Use getattr on the actual module to avoid Pylance errors
    cuda_version: Optional[str] = getattr(sys.modules["torch.version"], "cuda", None)
    assert cuda_version is not None, "❌ PyTorch installed without CUDA support"

    assert torch.backends.cudnn.is_available(), "❌ cuDNN backend not available"

    print(f"✅ CUDA version: {cuda_version}")
