import os
import numpy as np
import matplotlib.pyplot as plt


def main(file_path):
    data = np.load(file_path)
    rewards = data['reward']

    print(f"\n📁 Rollout: {os.path.basename(file_path)}")
    print(f"Total timesteps: {len(rewards)}")
    print("Step | Reward")

    for t, r in enumerate(rewards):
        mark = "🟢" if r > 0 else "   "
        print(f"{t:4d} | {r:7.3f} {mark}")


if __name__ == "__main__":
    main("data/car_racing/test/thread_0/rollout_3.npz")

