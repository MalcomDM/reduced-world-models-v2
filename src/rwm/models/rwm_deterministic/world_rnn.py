import torch
import torch.nn as nn

from rwm.config.config import PRNN_HIDDEN_DIM, WRNN_HIDDEN_DIM

class WorldRNN(nn.Module):
    def __init__(self, action_dim: int, dropout_prob: float = 0.8):
        super().__init__()
        # self.rnn_cell = nn.GRUCell(PRNN_HIDDEN_DIM + action_dim, WRNN_HIDDEN_DIM)
        self.rnn_cell = nn.LSTMCell(PRNN_HIDDEN_DIM + action_dim, WRNN_HIDDEN_DIM)
        self.reward_head = nn.Linear(WRNN_HIDDEN_DIM, 1)
        self.dropout_prob = dropout_prob


    def forward(
        self,
        h_prev: torch.Tensor,       # (B, WRNN_HIDDEN_DIM)
        c_prev: torch.Tensor,       # (B, WRNN_HIDDEN_DIM)
        x_spatial: torch.Tensor,    # (B, PRNN_HIDDEN_DIM)
        a_prev: torch.Tensor,       # (B, action_dim)
        force_keep_input: bool = False
    ):
        if force_keep_input:
            x_in = x_spatial
        else:
            B, D = x_spatial.shape
            # Generar máscara (B, 1) con Bernoulli: 1 = conservar, 0 = poner a cero
            mask = (torch.rand(B, 1, device=x_spatial.device) > self.dropout_prob).float()
            mask = mask.expand(-1, D)   # Expandir la máscara a (B, D)
            x_in = x_spatial * mask     

        h_new, c_new = self.rnn_cell(torch.cat((x_in, a_prev), dim=-1), (h_prev, c_prev))    # Concatenar [x_in, a_prev] y actualizar el GRUCell
        r_pred = self.reward_head(h_new)                                    # Predecir recompensa a partir de h_new

        return h_new, c_new, r_pred