import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CustomEnv(gym.Env):
    # def __init__(self, game, actions, minObsVal, maxObsVal, numObs, obsType):
    def __init__(self, game):
        super().__init__()
        self.game = game

        self.action_space = spaces.Discrete(self.game.actions)

        self.observation_space = spaces.Box(low=self.game.minObsVal, high=self.game.maxObsVal,
                                            shape=(self.game.numObs,), dtype=self.game.obsType)
        
    def step(self, action):
        self.done, reward, obs = self.game.step(action)
        
        self.info = {}
        truncated = False
        return obs, reward, self.done, truncated, self.info
    
    def reset(self, seed=None, options=None):
        self.done = False
        obs = self.game.reset()
        
        self.info = {}
        return obs, self.info