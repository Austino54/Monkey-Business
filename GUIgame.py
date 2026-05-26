import threading
import pygame
import time
from cells import Player, Cell
import trainGUI
import numpy as np
import math


def _reward_values(cfg: dict) -> dict:
    return {
        item.get("id"): item.get("value")
        for item in cfg.get("reward_items", [])
        if item.get("id") is not None
    }


def reward(data: np.ndarray, cfg: dict | None = None) -> float:
    if cfg is None:
        cfg = trainGUI.load_reward_config()
    vals = _reward_values(cfg)

    if data.size < 5:
        raise ValueError(
            "reward() needs at least 5 elements: "
            "[goal_x, goal_y, player_x, player_y, prev_dist]"
        )

    goal_x, goal_y = float(data[0]), float(data[1])
    player_x, player_y = float(data[2]), float(data[3])
    prev_dist = float(data[4])

    curr_dist = math.sqrt((player_x - goal_x) ** 2 + (player_y - goal_y) ** 2)
    rewPoints = 0.0

    if curr_dist < prev_dist:
        rewPoints += vals.get("GOT_CLOSER_PTS", 10)
    if curr_dist > prev_dist:
        rewPoints += vals.get("GOT_FARTHER_PTS", -10)
    if curr_dist <= vals.get("GOAL_TOLERANCE", 1.0):
        rewPoints += vals.get("REACHED_GOAL_PTS", 100)
    rewPoints += vals.get("DIST_BONUS_SCALE", 25) / (curr_dist + 1.0)

    return rewPoints

GREY = (140,140,140)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
    
class Game():
    actions = 4     # Number of actions the agent can take (currently discrete only)
    minObsVal = 0   # Minimum number that any observation value can be
    maxObsVal = 4   # Maximum number that any observation value can be
    numObs = 4      # Number of values in the observation space, i.e. how many values you will send the RL model (in this case 4 for player.x, player.y, goalx, and goaly)
    obsType = np.int64    # Data type of the observations, from numpy
    def __init__(self, h=5, w=5):
        pygame.init()
        self.width = w
        self.height = h
        self.screen = pygame.display.set_mode((self.width*100, self.height*100))

        self.grid = [[Cell(j, i, self.screen) for i in range(self.height)] for j in range(self.width)]

        self.goalx = 0
        self.goaly = 4
        self.grid[self.goalx][self.goaly].colorChange(self.screen, RED)
        self.player = Player(2, 2, self.screen, BLUE)
        self.running = True

    # Accept integer var action
    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        if action == -1:
            self.running = False
            return self.running
        
        dx = abs(self.player.x - self.goalx)
        dy = abs(self.player.y - self.goaly)

        if action == 0 and self.player.x>0:
            self.running = self.player.move(self.screen, self.grid[self.player.x-1][self.player.y])
        elif action == 1 and self.player.x<self.width-1:
            self.running = self.player.move(self.screen, self.grid[self.player.x+1][self.player.y])
        elif action == 2 and self.player.y>0:
            self.running = self.player.move(self.screen, self.grid[self.player.x][self.player.y-1])
        elif action == 3 and self.player.y<self.height-1:
            self.running = self.player.move(self.screen, self.grid[self.player.x][self.player.y+1])

        pygame.display.update()
        obs = np.array([self.player.x, self.player.y, self.goalx, self.goaly])
        
        # if not self.running:
        #     reward = 100
        # elif dx > abs(self.player.x - self.goalx) or dy > abs(self.player.y - self.goaly):
        #     reward = 10
        # else:
        #     reward = -20
        cfg = trainGUI.load_reward_config()
        rew = reward(np.append(obs, math.sqrt(dx**2 + dy**2)), cfg=cfg)
        
        return not self.running, rew, obs
    
    def reset(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width*100, self.height*100))

        self.grid = [[Cell(j, i, self.screen) for i in range(self.height)] for j in range(self.width)]

        self.goalx = 0
        self.goaly = 4
        self.grid[self.goalx][self.goaly].colorChange(self.screen, RED)
        self.player = Player(2, 2, self.screen, BLUE)
        self.running = True

        return np.array([self.player.x, self.player.y, self.goalx, self.goaly])

    def quit(self):
        pygame.quit()