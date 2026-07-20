import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


from app.utils.callbacks import CheckpointCallback, TimerCallback
from app.utils.json_stats_callbacks import EpochStatsCallback
from app.models.reduced_world_model import ReducedWorldModel
from app.datasets.windowed_dataset import WindowedDataset
from app.trainers.rwm_trainer import ReducedWorldModelTrainer
from app.config import BATCH_SIZE, LEARNING_RATE, NUM_WORKERS, ACTION_DIM

def main():
    parser = argparse.ArgumentParser(description="Train ReducedWorldModel")
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    parser.add_argument("--run_dir",    type=str,   default="runs/exp0")
    parser.add_argument("--logs_dir",    type=str,   default="runs/rwn/logs/train_stats.jsonl")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    # 1) Dataset y DataLoader
    dataset = WindowedDataset(attrs=('obs', 'action', 'reward'))
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    # 2) Crear modelo, optimizador y criterio
    model = ReducedWorldModel(action_dim=ACTION_DIM, dropout_prob=0.6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    # 3) Callbacks (ejemplo; define los tuyos según necesites)
    ckpt_cb = CheckpointCallback(model, folder=os.path.join(args.run_dir, "weights"), monitor="loss", mode="min")
    timer_cb = TimerCallback()
    json_cb = EpochStatsCallback(filepath=args.logs_dir)

    # 4) Instanciar el trainer, inyectando el DataLoader
    trainer = ReducedWorldModelTrainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        callbacks=[timer_cb, ckpt_cb, json_cb]
    )

    # 5) Llamar a fit(pasando solo epochs)
    trainer.fit(epochs=args.epochs)

    # 6) Guardar checkpoint final
    final_ckpt = os.path.join(args.run_dir, "weights", f"final_epoch{args.epochs}.pt")
    trainer.save_model(final_ckpt)
    print(f"Checkpoint saved to {final_ckpt}")

if __name__ == "__main__":
    main()