import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ws_server import ws_server

class CustomEnv(gym.Env):
    def __init__(self, server, actions, minObsVal, maxObsVal, numObs, obsType):
        super().__init__()

        self.action_space = spaces.Discrete(actions)

        self.observation_space = spaces.Box(low=minObsVal, high=maxObsVal,
                                            shape=(numObs,), dtype=obsType)