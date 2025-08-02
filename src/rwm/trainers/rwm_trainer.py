import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.config import M_WARMUP, WRNN_HIDDEN_DIM, ACTION_DIM
from app.models.reduced_world_model import ReducedWorldModel
from app.trainers.base_trainer import BaseTrainer


class ReducedWorldModelTrainer(BaseTrainer):

    def train( self ):
        running_loss = 0.0
        self.model.train()

        for batch_idx, (images_seq, actions_seq, rewards_seq) in enumerate(self.train_loader):
            images_seq = images_seq.to(self.device)      # images_seq: (B, T, 3, 64, 64)
            actions_seq = actions_seq.to(self.device)    # actions_seq: (B, T, action_dim)
            rewards_seq = rewards_seq.to(self.device)    # rewards_seq: (B, T, 1)

            B, T, _, _, _ = images_seq.shape
            self.optimizer.zero_grad()

            # 1) Inicializar hidden y acción anterior
            h_t = torch.zeros(B, WRNN_HIDDEN_DIM, device=self.device)
            c_t = torch.zeros(B, WRNN_HIDDEN_DIM, device=self.device)
            a_t_prev = torch.zeros(B, ACTION_DIM, device=self.device)
            loss_seq = 0.0

            # 2) Recorrer la secuencia paso a paso
            for t in range(T):
                img_t = images_seq[:, t]        # (B, 3, 64, 64)
                r_true_t = rewards_seq[:, t]    # (B, 1)

                force_keep = (t < M_WARMUP)

                # Forward completo: encoder → tokens → attention → patch_rnn → world_rnn
                h_t, c_t, r_pred_t, _, _ = self.model(
                    img=img_t,
                    a_prev=a_t_prev,
                    h_prev=h_t,
                    c_prev=c_t,
                    force_keep_input=force_keep
                )

                loss_seq += self.criterion(r_pred_t, r_true_t)  # Acumular MSE en todos los pasos
                a_t_prev = actions_seq[:, t]                    # Actualizar la acción “anterior”

            
            # 3) Dividir por T para obtener la pérdida promedio de la secuencia
            loss_batch = loss_seq / T
            loss_batch.backward()
            self.optimizer.step()

            running_loss += loss_batch.item()

            # 4) Disparar callbacks on_batch_end tras procesar este batch (una secuencia)
            for cb in self.callbacks:
                if hasattr(cb, 'on_batch_end'):
                    cb.on_batch_end(self, loss_batch.item())

        # 5) Pérdida promedio de la época (promedio sobre todos los batches)
        avg_loss = running_loss / len(self.train_loader)
        return avg_loss