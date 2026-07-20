import os, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from app.config import BATCH_SIZE, SAMPLE_BATCH, NUM_WORKERS, LEARNING_RATE, WEIGHT_DECAY
from app.datasets.frame_dataset import FrameDataset
from app.models.image_reconstructor import ImageReconstructor
from app.trainers.imgr_trainner import ImageTrainer
from app.utils.callbacks import CheckpointCallback, SaveReconCallback, TimerCallback


def main():
    parser = argparse.ArgumentParser(description="Train Reduced World-Model Reconstructor")
    parser.add_argument("--batch_size", type=int,     default=BATCH_SIZE,       help="Training batch size")
    parser.add_argument("--epochs",     type=int,     default=100,              help="Number of epochs")
    parser.add_argument("--lr",         type=float,   default=LEARNING_RATE,    help="Learning rate")
    parser.add_argument("--run_dir",    type=str,     default="runs/exp0",      help="Directory to save weights & viz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.run_dir,        exist_ok=True)
    os.makedirs(os.path.join(args.run_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(args.run_dir, "viz"),     exist_ok=True)

    # Dataset & DataLoader
    dataset      = FrameDataset()
    train_loader = DataLoader(dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              drop_last=True)

    model     = ImageReconstructor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # Callbacks
    ckpt_cb = CheckpointCallback(model, folder=os.path.join(args.run_dir, "weights"), monitor="loss", mode="min")
    recon_cb = SaveReconCallback(model, train_loader, folder=os.path.join(args.run_dir, "viz"), n_images=SAMPLE_BATCH)
    timer_cb = TimerCallback()

    trainer = ImageTrainer(model=model,
                      train_loader=train_loader,
                      optimizer=optimizer,
                      criterion=criterion,
                      device=device,
                      callbacks=[timer_cb, ckpt_cb, recon_cb])

    trainer.fit(epochs=args.epochs)

    # Guardar checkpoint final
    final_ckpt = os.path.join(args.run_dir, "weights", f"final_epoch{args.epochs}.pt")
    trainer.save_model(final_ckpt)
    print(f"Checkpoint saved to {final_ckpt}")

if __name__ == "__main__":
    main()