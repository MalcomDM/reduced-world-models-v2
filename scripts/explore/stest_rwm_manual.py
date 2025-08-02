import os, sys, argparse, torch, pygame
import numpy as np
from PIL import Image
import gymnasium as gym
import matplotlib.pyplot as plt

from app.envs.config import get_human_action
from app.models.reduced_world_model import ReducedWorldModel
from app.config import ACTION_DIM, WRNN_HIDDEN_DIM, INPUT_DIM, M_WARMUP


def load_model(checkpoint_path: str, device: torch.device) -> ReducedWorldModel:
    model = ReducedWorldModel(action_dim=ACTION_DIM, dropout_prob=0.85)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_name",  type=str,   default="CarRacing-v3")
    p.add_argument("--ckpt",      type=str,   default="runs/rwm/test_lstm/weights/best_loss_0.3011.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(args.ckpt):
        print(f"ERROR: no existe {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    model = load_model(args.ckpt, device)
    env = gym.make(args.env_name, render_mode="human", continuous=True)
    obs, _ = env.reset()

    pygame.init()
    clock = pygame.time.Clock()

    plt.ion()
    fig, ax = plt.subplots(figsize=(6,3))
    ax.set_xlabel("Paso")
    ax.set_ylabel("Recompensa")
    ax.set_title("Real vs Predicha (live)")
    line_true, = ax.plot([], [], label="Real")
    line_pred, = ax.plot([], [], label="Predicha")
    ax.legend()
    plt.show()

    h_t = torch.zeros(1, WRNN_HIDDEN_DIM, device=device)
    c_t = torch.zeros(1, WRNN_HIDDEN_DIM, device=device)
    a_prev = torch.zeros(1, ACTION_DIM, device=device)
    r_true_list, r_pred_list = [], []
    t = 0
    running = True

    while running:
        action, running = get_human_action()

        frame = Image.fromarray(obs).resize(INPUT_DIM[:2])
        frame = np.array(frame, dtype=np.float32) / 255.0
        x_t = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).to(device)
        force_keep = (t < M_WARMUP)
        with torch.no_grad():
            h_t, c_t, r_pred_t, _, _ = model(img=x_t, a_prev=a_prev, h_prev=h_t, c_prev=c_t, force_keep_input=force_keep)
        r_pred = float(r_pred_t)

        next_obs, r_true, terminated, truncated, _ = env.step(action)

        r_true_list.append(r_true)
        r_pred_list.append(r_pred)

        xs = np.arange(len(r_true_list))
        line_true.set_data(xs, r_true_list)
        line_pred.set_data(xs, r_pred_list)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

        obs = next_obs
        a_prev = torch.from_numpy(action).unsqueeze(0).to(device)
        t += 1

        if terminated or truncated:
            obs, _ = env.reset()
            r_true_list.clear(); r_pred_list.clear(); t = 0
            h_t.zero_(); c_t.zero_()

        clock.tick(60)

    # --------------------
    #  Cierre
    # --------------------
    env.close()
    pygame.quit()
    plt.ioff()
    plt.show()  # para dejar la ventana al salir

if __name__ == "__main__":
    main()
