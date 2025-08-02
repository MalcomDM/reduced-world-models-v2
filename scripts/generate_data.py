import os, argparse, threading
import numpy as np
from os.path import join
from PIL import Image

from app.config import INPUT_DIM
from app.envs.env import make_env
from app.envs.config import generate_smooth_action, generate_blended_action

def generate_rollouts(thread_id, env_name, data_dir, total_episodes, time_steps, action_refresh_rate):
    thread_data_dir = join(data_dir, f"thread_{thread_id}")        
    os.makedirs(thread_data_dir, exist_ok=True)

    print(f"Generating data for env {env_name}, {total_episodes} episodes")
    env = make_env(env_name)
    # env = make_env(env_name, 'human')

    for i in range(total_episodes):
        obs, _ = env.reset()
        obs_sequence, act_sequence, rew_sequence = [], [], []
        done, trunc = False, False
        act = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        t = 0

        early_push = 60
        no_improvement_steps = 0
        max_no_improvement_steps = 100  # ajustable
        best_post_rew = -0.1
        bad_episode = False
        last_positive_step = -1

        while t < time_steps and not (done or trunc or bad_episode):
            if t < early_push:
                act = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            elif t % action_refresh_rate == 0:
                act = generate_blended_action(act)
                
            obs = Image.fromarray(obs).resize(INPUT_DIM[:2])
            obs = np.array(obs).astype('float32') / 255.

            obs_sequence.append(obs)
            act_sequence.append(act)

            obs, rew, done, trunc, _ = env.step(act)
            rew_sequence.append(rew)

            if t > early_push:
                if rew > 0: last_positive_step = t
                if rew > best_post_rew:
                    best_post_rew = rew
                    no_improvement_steps = 0
                else:
                    no_improvement_steps += 1
                if no_improvement_steps >= max_no_improvement_steps:
                    bad_episode = True
            t += 1
        
        print(f"[Thread {thread_id}] Episode {i+1} finished after {t} timesteps.")
        # Se guarda hasta 30 pasos después del último reward positivo
        # para incluir contexto útil, incluso si el coche deja de avanzar
        if last_positive_step > 0:
            cutoff = min(len(obs_sequence), last_positive_step + 30)
            file_path = join(thread_data_dir, f'rollout_{i+1}.npz')
            np.savez_compressed(file_path,
                obs=np.array( obs_sequence[:cutoff] ),
                action=np.array( act_sequence[:cutoff] ),
                reward=np.array( rew_sequence[:cutoff]) )
        else:
            print(f"[Thread {thread_id}] Episode {i+1} not saved (no positive reward)")


def main(env_name, data_dir, total_episodes, time_steps, action_refresh_rate, num_threads):
    print(f"Generating data for env {env_name} with {num_threads} threads")
    os.makedirs(data_dir, exist_ok=True)
    episodes_per_thread = total_episodes // num_threads
    threads = []

    for i in range(num_threads):
        t = threading.Thread(target=generate_rollouts, args=(i, env_name, data_dir, episodes_per_thread, time_steps, action_refresh_rate))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Create new training data'))
    parser.add_argument('--env_name', type=str, default='car_racing')
    parser.add_argument('--data_dir', type=str, default='app/data/car_racing/test/')
    parser.add_argument('--total_episodes', type=int, default=200,
                        help='episodes to generate per worker')
    parser.add_argument('--time_steps', type=int, default=1000,
                        help='how many timesteps at start of episode?')
    parser.add_argument('--action_refresh_rate', default=5, type=int,
                        help='how often to change the random action, in frames')
    parser.add_argument('--num_threads', type=int, default=8)

    args = parser.parse_args()
    main(args.env_name, args.data_dir, args.total_episodes, args.time_steps, args.action_refresh_rate, args.num_threads)