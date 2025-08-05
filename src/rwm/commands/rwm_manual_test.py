import sys, typer, csv
import torch, pygame
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from numpy.typing import NDArray
from typing import Any

import gymnasium as gym
from gymnasium.core import Env

from rwm.models.rwm_deterministic.model import ReducedWorldModel
from rwm.policies.human_policy import HumanPolicy
from rwm.config.config import ACTION_DIM, WRNN_HIDDEN_DIM, INPUT_DIM, M_WARMUP


def load_model(checkpoint_path: str, device: torch.device) -> ReducedWorldModel:
	model = ReducedWorldModel(action_dim=ACTION_DIM, dropout_prob=0.85)
	ckpt = torch.load(checkpoint_path, map_location=device)
	state = ckpt.get("model_state", ckpt)
	model.load_state_dict(state)
	return model.to(device).eval()


def preprocess_obs(obs: NDArray[np.uint8], device: torch.device) -> torch.Tensor:
	img = Image.fromarray(obs).resize(INPUT_DIM[:2])
	img = np.array(img, dtype=np.float32) / 255.0
	tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # type: ignore (1, C, H, W)
	return tensor.to(device)


app = typer.Typer()


def main(ckpt: Path, env_name: str):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if not ckpt.exists():
		print(f"❌ ERROR: checkpoint not found at {str(ckpt)}", file=sys.stderr)
		sys.exit(1)

	model = load_model(str(ckpt), device)
	env: Env[Any, Any] = gym.make(env_name, render_mode="human", continuous=True)	# type: ignore 
	obs, _ = env.reset()

	policy = HumanPolicy()
	policy.reset()
	pygame.init()
	clock = pygame.time.Clock()

	# Plotting
	plt.ion()										# type: ignore 
	fig, ax = plt.subplots(figsize=(6, 3))			# type: ignore 
	ax.set_xlabel("Step")							# type: ignore 
	ax.set_ylabel("Reward")							# type: ignore 
	ax.set_title("Real vs Predicted Reward")		# type: ignore 
	line_true, = ax.plot([], [], label="True")		# type: ignore 
	line_pred, = ax.plot([], [], label="Pred")		# type: ignore 
	ax.legend()										# type: ignore 
	plt.show()										# type: ignore 

	# World Model state
	h_t = torch.zeros(1, WRNN_HIDDEN_DIM, device=device)
	c_t = torch.zeros(1, WRNN_HIDDEN_DIM, device=device)
	a_prev = torch.zeros(1, ACTION_DIM, device=device)

	r_true_list: list[float] = []
	r_pred_list: list[float] = []
	t = 0

	while policy.is_running():
		action = policy.act(obs)
		frame_tensor = preprocess_obs(obs, device)

		force_keep = t < M_WARMUP
		with torch.no_grad():
			h_t, c_t, r_pred_t, *_ = model(img=frame_tensor, a_prev=a_prev, h_prev=h_t, c_prev=c_t, force_keep_input=force_keep)
		r_pred = float(r_pred_t.item())

		next_obs, r_true, terminated, truncated, _ = env.step(action)

		r_true_list.append(float(r_true))
		r_pred_list.append(r_pred)

		xs = np.arange(len(r_true_list))
		line_true.set_data(xs, r_true_list)
		line_pred.set_data(xs, r_pred_list)
		ax.relim()
		ax.autoscale_view()
		fig.canvas.draw()														# type: ignore 
		fig.canvas.flush_events()

		obs = next_obs
		a_prev = torch.from_numpy(action).unsqueeze(0).to(device)				# type: ignore 
		t += 1

		if terminated or truncated:
			obs, _ = env.reset()
			h_t.zero_(); c_t.zero_()
			t = 0
			r_true_list.clear()
			r_pred_list.clear()

		clock.tick(60)

	if r_true_list:
		save_reward_log(Path("reward_log.csv"), r_true_list, r_pred_list)

	# Cleanup
	env.close()
	pygame.quit()
	plt.ioff()					# type: ignore 
	plt.show()					# type: ignore


def save_reward_log(path: Path, r_true_list: list[float], r_pred_list: list[float]) -> None:
    assert len(r_true_list) == len(r_pred_list)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "r_true", "r_pred"])
        for i, (rt, rp) in enumerate(zip(r_true_list, r_pred_list)):
            writer.writerow([i, rt, rp])
    print(f"✅ Saved reward log to {path}")


@app.command()
def run(
    ckpt: Path = typer.Argument(..., exists=True, help="Path to the trained world model checkpoint"),
    env_name: str = typer.Option("CarRacing-v3", help="Gymnasium environment name")
) -> None:
	main(ckpt, env_name)



if __name__ == "__main__":
	app()
