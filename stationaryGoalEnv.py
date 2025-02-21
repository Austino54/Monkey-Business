import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import time
from cells import Cell
from cells import Player

GREY = (140,140,140)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)

class StationaryGoalEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, width=5, height=5):
        super().__init__()
        self.width = width
        self.height = height
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=4,
                                            shape=(4,), dtype=np.int64)

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.done = True
                break

        # self.cooldown = 0
        # if time.time() >= self.cooldown:
            # self.cooldown = time.time() + 0.5
            # pygame.time.delay(10)
        if action == 0 and self.player.x>0:
            self.done = not self.player.move(self.screen, self.grid[self.player.x-1][self.player.y])
        elif action == 1 and self.player.x<self.width-1:
            self.done = not self.player.move(self.screen, self.grid[self.player.x+1][self.player.y])
        elif action == 2 and self.player.y>0:
            self.done = not self.player.move(self.screen, self.grid[self.player.x][self.player.y-1])
        elif action == 3 and self.player.y<self.height-1:
            self.done = not self.player.move(self.screen, self.grid[self.player.x][self.player.y+1])
        # else:
            # self.cooldown = 0
        pygame.display.update()

        if self.done:
            self.reward = 100
        else:
            self.reward = -10
            
        self.observation = np.array([self.player.x, self.player.y, 0, self.height-1])

        self.info = {}
        truncated = False
        return self.observation, self.reward, self.done, truncated, self.info

    def reset(self, seed=None, options=None):
        self.done = False
        self.reward = 0
        pygame.init()
        # self.width = 5
        # self.height = 5
        self.screen = pygame.display.set_mode((self.width*100, self.height*100))

        self.grid = [[Cell(j, i, self.screen) for i in range(self.height)] for j in range(self.width)]
        self.grid[0][self.height-1].colorChange(self.screen, RED)
        self.player = Player(2, 2, self.screen, BLUE)

        # player_x, player_y, goal_x(?), goal_y(?)
        self.observation = np.array([self.player.x, self.player.y, 0, self.height-1])

        self.info = {}
        return self.observation, self.info

    # def render(self):
    #     ...

    def close(self):
        pygame.quit()