import pytest
import torch
from torch import Tensor

from rwm.models.rwm_deterministic.encoder import Encoder
from rwm.config.config import INPUT_DIM


@pytest.mark.models
def test_encoder_output_shape() -> None:
	encoder = Encoder()
	encoder.eval()

	batch_size = 4
	h, w, c = INPUT_DIM
	x: Tensor = torch.rand(batch_size, c, h, w)

	with torch.no_grad():
		output: Tensor = encoder(x)

	assert output.ndim == 4, "Output must be (B, C_out, H_out, W_out)"
	b, c_out, h_out, w_out = output.shape

	assert b == batch_size, f"Batch size mismatch: got {b}, expected {batch_size}"
	assert c_out > 0, "Output channels must be > 0"
	assert h_out > 0 and w_out > 0, "Output spatial dims must be > 0"


@pytest.mark.models
def test_encoder_is_deterministic() -> None:
	encoder = Encoder()
	encoder.eval()

	h, w, c = INPUT_DIM
	x: Tensor = torch.rand(1, c, h, w)
	with torch.no_grad():
		out1 = encoder(x)
		out2 = encoder(x)

	assert torch.allclose(out1, out2), "Encoder output should be deterministic in eval mode"



