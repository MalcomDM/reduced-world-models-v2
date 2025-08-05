from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from PIL import Image

import torch
from torch import Tensor


INPUT_SIZE: Tuple[int, int] = (64, 64)	# target size should match your model’s expected input

_RESAMPLE = Image.Resampling.BILINEAR


def preprocess_obs(obs: NDArray[np.uint8]) -> Tensor:
    """
    Preprocess a single RGB observation for the world model.

    Args:
        obs: array of shape (H, W, 3), dtype=uint8, values in [0,255]

    Returns:
        Tensor of shape (1, 3, 64, 64), dtype=float32, values in [0,1]
    """
    # 1. PIL resize
    img = Image.fromarray(obs)
    img = img.resize(INPUT_SIZE, resample=_RESAMPLE)  # type: ignore[arg-type]

    # 2. To NumPy float32 [0,1]
    arr: NDArray[np.float32] = np.array(img, dtype=np.float32) / 255.0

    # 3. To tensor, permute to (C, H, W), add batch dim
    tensor: Tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)	# type: ignore

    return tensor
