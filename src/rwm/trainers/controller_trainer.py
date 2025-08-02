import os, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from torch.utils.data import DataLoader
from app.models.reduced_world_model import ReducedWorldModel
from app.models.controller import Controller
from app.datasets.windowed_dataset import WindowedDataset
from app.config import ACTION_DIM, WRNN_HIDDEN_DIM, OBSERVATIONAL_DROPOUT

def simulate_rollout(rwm, ctrl, init_frame, horizon):
    device = next(rwm.parameters()).device
    rwm.eval(); ctrl.eval()

    # estados
    h = torch.zeros(1, WRNN_HIDDEN_DIM, device=device)
    c = torch.zeros(1, WRNN_HIDDEN_DIM, device=device)
    a_prev = torch.zeros(1, ACTION_DIM, device=device)

    # primer paso: extraer h_spatial de init_frame
    with torch.no_grad():
        feat   = rwm.encoder(init_frame)
        toks   = rwm.tokenizer(feat)
        logits = rwm.scorer(toks)
        mask, idxs = rwm.selector(logits)
        h_spat = rwm.patch_rnn(toks, idxs)

    rewards = []
    for t in range(horizon):
        x_in = h_spat if t == 0 else torch.zeros_like(h_spat)
        with torch.no_grad():
            h, c, r_pred, _, _ = rwm.world_rnn(
                h_prev=h, c_prev=c,
                x_spatial=x_in,
                a_prev=a_prev,
                force_keep_input=(t==0)
            )
        a = ctrl(h)
        rewards.append(r_pred.item())
        a_prev = a
    return rewards

class ControllerTrainer:
    def __init__(self,
                 rwm_ckpt: str,
                 action_fn,
                 batch_size:   int = 1,
                 horizon:      int = 20,
                 n_rollouts:   int = 32,
                 device:       str = "cuda",
                 k_best:       int = 8,
                 lr:           float = 5e-4):
        # 1) cargar World Model
        self.rwm = ReducedWorldModel(dropout_prob=OBSERVATIONAL_DROPOUT).to(self.device)
        self.rwm.load_state_dict(torch.load(rwm_ckpt, map_location=self.device))
        self.rwm.eval()
        # 2) controller
        self.ctrl = Controller().to(self.device)
        self.opt  = optim.Adam(self.ctrl.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        # 3) datos iniciales
        base_ds = WindowedDataset(attrs=('obs','action','reward'))
        # tomamos solo el primer frame de cada ventana
        class InitDS(torch.utils.data.Dataset):
            def __init__(self, base): self.base = base
            def __len__(self): return len(self.base)
            def __getitem__(self,i):
                obs_seq,_,_ = self.base[i]
                return obs_seq[0]
        self.loader = DataLoader(InitDS(base_ds),
                                 batch_size=batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True)

        self.action_fn = action_fn
        self.horizon   = horizon
        self.n_rollouts= n_rollouts
        self.k_best    = k_best

    def train(self, epochs: int = 20):
        for ep in range(epochs):
            tot_loss = 0.0
            for frames in self.loader:
                init = frames.to(self.device).unsqueeze(0)  # (1,3,64,64)
                # 1) simular N rollouts
                trajs = []
                for _ in range(self.n_rollouts):
                    r = simulate_rollout(self.rwm, self.ctrl, init,
                                         self.horizon, self.action_fn)
                    trajs.append((sum(r), r))
                # 2) top-K
                trajs.sort(key=lambda x: x[0], reverse=True)
                best = trajs[:self.k_best]
                # 3) construir batch states ↔ actions
                Hs, As = [], []
                for total, rewards in best:
                    # re-simular para recuperar acciones/hiddens
                    # o bien guardar ambas listas dentro de simulate_rollout
                    pass  # similar a lo anterior
                # 4) forward & loss sobre Controller
                #    ctrl(Hs) vs As -> MSE, backward, step
                # ...
            print(f"[Epoch {ep+1}/{epochs}] loss {tot_loss:.4f}")
        # 5) guardar
        torch.save(self.ctrl.state_dict(), "runs/controller_final.pt")