import os, torch, argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from app.envs.env import make_env
from app.envs.config import generate_smooth_action, generate_blended_action

from app.models.reduced_world_model import ReducedWorldModel
from app.config import ACTION_DIM, M_WARMUP, INPUT_DIM, WRNN_HIDDEN_DIM


def load_model(checkpoint_path: str, device: torch.device) -> ReducedWorldModel:
    model = ReducedWorldModel(action_dim=ACTION_DIM, dropout_prob=0.85)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()
    return model

def run_episode(model: ReducedWorldModel, device: torch.device, env_name:str="", max_steps: int = 300):
    env = make_env(env_name)
    obs, _ = env.reset()
    obs = Image.fromarray(obs).resize(INPUT_DIM[:2])
    obs = np.array(obs, dtype=np.float32) / 255.0  # normalizar a [0,1]
    h_t = torch.zeros(1, WRNN_HIDDEN_DIM, device=device)
    a_prev = torch.zeros(1, ACTION_DIM, device=device)

    r_true_list = []
    r_pred_list = []

    done = False
    t = 0
    early_push = 60

    while not done and t < max_steps:
        # 1) Convertir frame a tensor (1, C, H, W)
        frame = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,64,64)

        # 2) Predict reward_rnn sin backprop
        force_keep = (t < M_WARMUP)
        with torch.no_grad():
            h_t, r_pred_t, _, _ = model(
                img=frame,
                a_prev=a_prev,
                h_prev=h_t,
                force_keep_input=force_keep
            )
        r_pred = r_pred_t.item()

        # 3) Elegir acción
        if t < early_push:
            act = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            act = generate_blended_action(act)
            act = act.astype(np.float32)

        # 4) Ejecutar step en el entorno
        next_obs, r_true, done, truncated, _ = env.step(act)
        obs = Image.fromarray(next_obs).resize(INPUT_DIM[:2])
        obs = np.array(obs, dtype=np.float32) / 255.0

        # 5) Registrar reward real y predicho
        r_true_list.append(r_true)
        r_pred_list.append(r_pred)

        # 6) Preparar a_prev y t++
        a_prev = torch.from_numpy(act).unsqueeze(0).to(device)  # (1,3)
        t += 1

    env.close()
    return r_true_list, r_pred_list


def save_reward_comparison(r_true, r_pred, max_steps=200, filepath="runs/rwm/comp/reward_comparison.png"):
    # Asegurar backend sin display
    plt.switch_backend('Agg')

    out_dir = os.path.dirname(filepath)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    steps = np.arange(len(r_true[:max_steps]))
    true_vals = np.array(r_true[:max_steps])
    pred_vals = np.array(r_pred[:max_steps])

    plt.figure(figsize=(6, 3))
    plt.plot(steps, true_vals, label='Recompensa real')
    plt.plot(steps, pred_vals, label='Predicción')
    plt.xlabel('Paso')
    plt.ylabel('Recompensa')
    plt.title('Comparativo: Recompensa real vs. Predicha')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Test Reduced World Model")
    parser.add_argument('--env_name',   type=str,   default='car_racing')
    parser.add_argument("--dict_path",  type=str,   default="runs/rwm/test1/weights/best_loss_0.3305.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = args.dict_path
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No se encontró {checkpoint_path}")

    # 1) Cargar modelo
    model = load_model(checkpoint_path, device)

    # 2) Ejecutar 1 episodio y obtener listas de recompensas
    r_true, r_pred = run_episode(model, device, args.env_name)

    # 3) Imprimir los primeros 200 pasos o todos si el episodio fue más corto
    N = min(200, len(r_true))
    print("\nPaso   Recompensa real    Predicción")
    print("----------------------------------------")
    for i in range(N):
        print(f"{i:3d}    {r_true[i]:8.4f}       {r_pred[i]:8.4f}")

    # 4) Opcional: calcular métricas globales de RMSE
    if len(r_true) > 0:
        import numpy as np
        mse = np.mean((np.array(r_true[:N]) - np.array(r_pred[:N]))**2)
        rmse = np.sqrt(mse)
        print(f"\nRMSE (primeros {N} pasos): {rmse:.4f}")

    # 5) Graficar la diferencia
    save_reward_comparison(r_true, r_pred, max_steps=200)

    

if __name__ == "__main__":
    main()