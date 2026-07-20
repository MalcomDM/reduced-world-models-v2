import pygame
import gymnasium as gym
from app.envs.config import get_human_action

# Inicializar entorno
env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
obs, _ = env.reset()
clock = pygame.time.Clock()

running = True

if __name__ == "__main__":
    while running:
        action, running = get_human_action()
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()
        clock.tick(60)

    env.close()