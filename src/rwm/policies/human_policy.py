import pygame
import numpy as np
from numpy.typing import NDArray
from typing import Optional

from .base_policy import BasePolicy


class HumanPolicy(BasePolicy):
    def __init__(self) -> None:
        pygame.init()
        self.running: bool = True


    def reset(self) -> None:
        self.running = True
        pygame.event.clear()


    def act(
        self,
        obs: NDArray[np.float32],
        prev_action: Optional[NDArray[np.float32]] = None
    ) -> NDArray[np.float32]:
        self._process_quit_events()

        action = np.zeros(3, dtype=np.float32)
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:		action[0] = -1.0
        if keys[pygame.K_RIGHT]:	action[0] = 1.0
        if keys[pygame.K_UP]:		action[1] = 1.0
        if keys[pygame.K_DOWN]:		action[2] = 0.8

        return action


    def _process_quit_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False


    def is_running(self) -> bool:
        return self.running