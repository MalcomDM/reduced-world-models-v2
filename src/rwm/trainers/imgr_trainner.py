from app.trainers.base_trainer import BaseTrainer


class ImageTrainer(BaseTrainer):


    def train(self):
        running_loss = 0.0
        self.model.train()

        for batch_idx, (x,y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)

            if isinstance(output, tuple):
                recon = output[0]
            else:
                recon = output
            loss = self.criterion(recon, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Aquí disparo on_batch_end con el valor flotante loss_item
            for cb in self.callbacks:
                if hasattr(cb, 'on_batch_end'):
                    cb.on_batch_end(self, loss.item())

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss


    def train_with_generator(self,
                             generator,
                             steps_per_epoch: int,
                             epochs: int,
                             initial_epoch: int = 0):

        for epoch in range(initial_epoch, epochs):
            self.epoch = epoch

            running_loss = 0.0
            self.model.train()
            gen_iter = iter(generator)
            for step in range(steps_per_epoch):
                try:
                    x, y = next(gen_iter)
                except StopIteration:
                    gen_iter = iter(generator)
                    x, y = next(gen_iter)

                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                if isinstance(output, tuple):
                    recon = output[0]
                else:
                    recon = output
                loss = self.criterion(recon, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / steps_per_epoch