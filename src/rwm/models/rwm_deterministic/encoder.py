from torch import nn
from torch import Tensor

from rwm.config.config import (
	IN_CHANNELS, CONV_FILTERS, CONV_KERNEL_SIZES,
    CONV_STRIDES, CONV_PADDINGS, CONV_ACTIVATIONS
)

ACT_MAP = {
    "relu": nn.ReLU,
    "lrelu": lambda: nn.LeakyReLU(0.2),
    "elu": nn.ELU,
}


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()				# type: ignore[reportUnknownMemberType]
        
        layers: list[nn.Module] = []
        c: int = IN_CHANNELS
        
        for filt, ks, st, p, act in zip(
            CONV_FILTERS, CONV_KERNEL_SIZES,
            CONV_STRIDES, CONV_PADDINGS, CONV_ACTIVATIONS
        ):
            layers += [
                nn.Conv2d(c, filt, kernel_size=ks, stride=st, padding=p),
                nn.BatchNorm2d(filt),
                ACT_MAP[act](),
            ]
            c = filt
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.encoder(x)