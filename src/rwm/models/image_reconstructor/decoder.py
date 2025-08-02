import torch
import torch.nn as nn

from app.config import PRNN_HIDDEN_DIM, DENSE_SIZE, CONV_T_FILTERS, CONV_T_KERNEL_SIZES, CONV_T_STRIDES, CONV_T_ACTIVATIONS

class Decoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        # project h → small spatial grid, e.g. 4×4×128
        self.fc = nn.Linear(PRNN_HIDDEN_DIM, DENSE_SIZE)
        layers = []
        in_ch = 128
        for out_ch, k, s, act in zip( CONV_T_FILTERS, CONV_T_KERNEL_SIZES, CONV_T_STRIDES, CONV_T_ACTIVATIONS):
            pad = 1
            out_pad = (k-s) % 2
            layers.append(nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=k, stride=s, padding=pad, output_padding=out_pad
            ))
            if act == 'relu': layers.append(nn.ReLU(inplace=True))
            elif act == 'sigmoid': layers.append(nn.Sigmoid())
            else: raise ValueError(f"Unsupported activation {act}")
            in_ch = out_ch
        self.deconv = nn.Sequential(*layers)


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        x = self.fc(z)                          # (B, 4*4*128)
        x: torch.Tensor = x.view(B, 128, 4, 4)  # (B, 128, 4, 4)
        x = self.deconv(x)                      # (B, 3, 64, 64)
        return x
    