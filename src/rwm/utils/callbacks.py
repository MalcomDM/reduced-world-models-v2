import os, time, pickle, torch
from torchvision.utils import make_grid, draw_bounding_boxes, save_image

from app.config import INPUT_DIM, FEATURE_MAP_SIZE, PATCH_SIZE, PATCH_STRIDE

class Callback:
    def on_epoch_begin(self, trainer): ...
    def on_batch_end(self, trainer, loss): ...
    def on_epoch_end(self, trainer, loss): ...


class CheckpointCallback(Callback):
    def __init__(self, model, folder, monitor='loss', mode='min'):
        os.makedirs(folder, exist_ok=True)
        self.model   = model
        self.folder  = folder
        self.monitor = monitor
        self.mode    = mode
        self.best    = float('inf') if mode=='min' else -float('inf')

    def on_epoch_end(self, trainer, loss):
        score = loss if self.monitor=='loss' else getattr(trainer, self.monitor)
        improved = (score < self.best) if self.mode=='min' else (score > self.best)
        if improved:
            self.best = score
            path = os.path.join(self.folder, f'best_{self.monitor}_{score:.4f}.pt')
            torch.save(self.model.state_dict(), path)
            print(f"[Checkpoint] Saved improved model to {path}")


class SaveReconCallback(Callback):
    def __init__(self, model, sample_loader, folder, n_images=4):
        os.makedirs(folder, exist_ok=True)
        self.model      = model
        self.sample_iter = iter(sample_loader)
        self.folder     = folder
        self.n_images   = n_images
        self._position_detection()

    def _position_detection(self):
        # precompute some constants for mapping patch → pixel coords
        self.H_in, self.W_in = INPUT_DIM[:2]
        self.Hf, self.Wf     = FEATURE_MAP_SIZE, FEATURE_MAP_SIZE
        # how much each feature‐map “pixel” covers in the original image
        self.scale_x = self.W_in / self.Wf
        self.scale_y = self.H_in / self.Hf

        # number of patches per row/col
        self.Ndim = ((self.Hf - PATCH_SIZE) // PATCH_STRIDE) + 1

    # def on_epoch_end(self, trainer, loss):
    #     self.model.eval()
    #     try:
    #         imgs, _ = next(self.sample_iter)
    #     except StopIteration:
    #         self.sample_iter = iter(self.sample_iter._dataset)
    #         imgs, _ = next(self.sample_iter)
    #     imgs = imgs[:self.n_images].to(next(self.model.parameters()).device)
    #     with torch.no_grad():
    #         recon, _, _ = self.model(imgs)
    #     # stack originals and recons
    #     grid = make_grid(torch.cat([imgs, recon]), nrow=self.n_images)
    #     save_image(grid, os.path.join(self.folder, f'recon_{trainer.epoch:03d}.png'))
    #     self.model.train()

    def on_epoch_end(self, trainer, loss):
        self.model.eval()

        try:
            imgs, _ = next(self.sample_iter)
        except StopIteration:
            # re-initialize sampler if exhausted
            self.sample_iter = iter(self.sample_iter._dataset)
            imgs, _ = next(self.sample_iter)

        imgs = imgs[:self.n_images].to(next(self.model.parameters()).device)

        with torch.no_grad():
            recon, mask, indices = self.model(imgs)

        # 1) Draw bounding boxes on the *original* images
        #   a) Convert floats [0,1] → uint8 [0,255]
        orig_uint8 = (imgs * 255).to(torch.uint8)
        bboxes_batch = []
        colors_batch = []

        for b in range(self.n_images):
            idxs = indices[b].tolist()     # list of K patch-indices
            boxes = []
            for idx in idxs:
                row = idx // self.Ndim
                col = idx %  self.Ndim
                # top-left in feature coords
                x1_f = col * PATCH_STRIDE
                y1_f = row * PATCH_STRIDE
                # scale up to original pixels
                x1 = x1_f * self.scale_x
                y1 = y1_f * self.scale_y
                w  = PATCH_SIZE * self.scale_x
                h  = PATCH_SIZE * self.scale_y
                # junction requires ints
                boxes.append([int(x1), int(y1), int(x1 + w), int(y1 + h)])
            bboxes_batch.append(torch.tensor(boxes, dtype=torch.int))
            # pick one color per box
            colors_batch.append(["blue"] * len(idxs))

        # draw_bounding_boxes expects (B,C,H,W) uint8, list of tensors
        overlaid = []
        for b in range(self.n_images):
            overlay = draw_bounding_boxes(
                image = orig_uint8[b],
                boxes = bboxes_batch[b],
                colors= colors_batch[b],
                width = 2
            )
            overlaid.append(overlay)
        overlaid = torch.stack(overlaid, dim=0)  # (n_images,3,H,W)

        # 2) build two grids:
        #    a) originals with boxes
        #    b) reconstructions
        grid_orig = make_grid(overlaid,    nrow=self.n_images)
        grid_recon= make_grid((recon*255).to(torch.uint8), nrow=self.n_images)

        device = recon.device
        grid_orig  = grid_orig.to(device).to(torch.float32).div_(255.0)
        grid_recon = grid_recon.to(device).to(torch.float32).div_(255.0)

        # 3) stack vertically and save
        both = torch.cat([grid_orig, grid_recon], dim=1)
        save_image(both, os.path.join(self.folder, f'recon_{trainer.epoch:03d}.png'))

        self.model.train()


class TimerCallback(Callback):
    def on_epoch_begin(self, trainer):
        trainer._start_time = time.time()
    def on_epoch_end(self, trainer, loss):
        elapsed = time.time() - trainer._start_time
        print(f"[Timer] Epoch {trainer.epoch} took {elapsed:.1f}s")