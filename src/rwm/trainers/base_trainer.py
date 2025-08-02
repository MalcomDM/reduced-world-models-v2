import os
import torch

class BaseTrainer:

    def __init__(self, model, train_loader, optimizer, criterion, device, callbacks=None):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.callbacks = callbacks or []
        self.epoch = 0

    def train(self):
        raise NotImplementedError("Implementar en subclases")


    def fit(self, epochs):
        for cb in self.callbacks:
            if hasattr(cb, 'on_batch_begin'):
                cb.on_batch_begin(self)

        for ep in range(epochs):
            self.epoch = ep
            for cb in self.callbacks:
                if hasattr(cb, 'on_epoch_begin'):
                    cb.on_epoch_begin(self)

            loss = self.train()

            for cb in self.callbacks:
                if hasattr(cb, 'on_epoch_end'):
                    cb.on_epoch_end(self, loss)


    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)


    def load_model(self, path: str, load_optimizer: bool = False):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.to(self.device)
        if load_optimizer and 'optimizer_state' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'epoch' in ckpt:
            self.epoch = ckpt['epoch']