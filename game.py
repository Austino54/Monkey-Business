import threading
import pygame
import time
from cells import Player, Cell
import numpy as np

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
            pass
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
        
        if not self.running:
            reward = 100
        elif dx > abs(self.player.x - self.goalx) or dy > abs(self.player.y - self.goaly):
            reward = 10
        else:
            reward = -20
        
        return not self.running, reward, obs
    
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