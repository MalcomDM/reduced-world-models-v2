import argparse, random, torch
import numpy as np

from app.models.reduced_world_model import ReducedWorldModel
from app.datasets.windowed_dataset import WindowedDataset
from app.config import ACTION_DIM, SEQ_LEN, WRNN_HIDDEN_DIM, M_WARMUP


def load_model(model: ReducedWorldModel, checkpoint_path: str, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state'] if 'model_state' in state else state)
    model.to(device)
    model.eval()
    return model


def sample_and_compare(model: ReducedWorldModel, dataset: WindowedDataset, device: torch.device, num_windows: int = 5):
    results = []
    for _ in range(num_windows):
        # Escoger un índice aleatorio en [0, len(dataset))
        idx = random.randrange(len(dataset))
        obs_seq, action_seq, reward_seq = dataset[idx]  # obs_seq: (SEQ_LEN, C, H, W)
                                                         # action_seq: (SEQ_LEN, action_dim)
                                                         # reward_seq: (SEQ_LEN, 1)
        # Llevar tensores a device y añadir batch dim
        obs_seq = obs_seq.unsqueeze(0).to(device)        # (1, SEQ_LEN, C, H, W)
        action_seq = action_seq.unsqueeze(0).to(device)  # (1, SEQ_LEN, action_dim)
        reward_seq = reward_seq.unsqueeze(0).to(device)  # (1, SEQ_LEN, 1)

        B = 1
        # Inicializar estado oculto y acción previa a ceros
        h_t = torch.zeros(B, WRNN_HIDDEN_DIM, device=device)
        a_prev = torch.zeros(B, ACTION_DIM, device=device)

        # Recorrer los SEQ_LEN pasos y predecir recompensa en cada uno
        r_pred_list = []
        for t in range(SEQ_LEN):
            frame = obs_seq[:, t]            # (1, C, H, W)
            r_true_t = reward_seq[:, t]      # solo para referencia; no lo usamos en forward
            force_keep = (t < M_WARMUP)

            # Forward paso t
            with torch.no_grad():
                h_t, r_pred_t, _, _ = model(
                    img=frame,
                    a_prev=a_prev,
                    h_prev=h_t,
                    force_keep_input=force_keep
                )
            r_pred_list.append(r_pred_t.item())

            # Preparar acción “real” del dataset para el siguiente paso
            a_prev = action_seq[:, t]

        # Convertir reward_seq y r_pred_list a vectores 1D en CPU
        r_true_vec = reward_seq.squeeze(0).squeeze(-1).cpu().numpy()   # (SEQ_LEN,)
        r_pred_vec = np.array(r_pred_list)                             # (SEQ_LEN,)

        results.append((r_true_vec, r_pred_vec))

    return results

def print_comparisons(results):
    for i, (r_true, r_pred) in enumerate(results, 1):
        print(f"\n=== Ventana {i} ===")
        print(f"{'Paso':>4}  {'Recompensa real':>15}  {'Predicción':>11}")
        print("-" * 38)
        for t in range(len(r_true)):
            print(f"{t:4d}  {r_true[t]:15.4f}  {r_pred[t]:11.4f}")

def main():
    parser = argparse.ArgumentParser(description="Test Reduced World Model")
    parser.add_argument("--dict_path",    type=str,   default="runs/rwm/test1/weights/best_loss_0.3305.pt")
    args = parser.parse_args()

    # Ruta al checkpoint preentrenado (ajusta según donde guardaste)
    checkpoint_path = args.dict_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Construir el modelo y cargar pesos
    model = ReducedWorldModel(action_dim=ACTION_DIM, dropout_prob=0.8)
    model = load_model(model, checkpoint_path, device)

    # 2) Crear el dataset de ventanas
    dataset = WindowedDataset(attrs=('obs', 'action', 'reward'))

    # 3) Tomar 5 ventanas aleatorias y obtener comparaciones
    comparisons = sample_and_compare(model, dataset, device, num_windows=5)

    # 4) Imprimir real vs. predicho para cada ventana
    print_comparisons(comparisons)

if __name__ == "__main__":
    main()
