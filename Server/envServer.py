# Run this file to start the server that trains a model

import asyncio
from ws_server import ws_server
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import threading
import os
import time
from stable_baselines3 import A2C
from stable_baselines3 import PPO

started = False
gotRewardObs = False
reward = 0
obs = [0]
done = False

dataTypes = {
    'np.int32': np.int32,
    'np.int64': np.int64,
    'np.uint32': np.uint32,
    'np.uint16': np.uint16,
    'np.float64': np.float64,
    'np.float32': np.float32
}

# Custom environment class
class CustomEnv(gym.Env):
    def __init__(self, actions, minObsVal, maxObsVal, numObs, obsType):
        super().__init__()

        self.action_space = spaces.Discrete(actions)

        self.observation_space = spaces.Box(low=minObsVal, high=maxObsVal,
                                            shape=(numObs,), dtype=dataTypes[obsType])
        
    def step(self, action):
        global gotRewardObs
        asyncio.run(getState(int(action)))
        
        gotRewardObs = False
        self.info = {}
        truncated = False
        return obs, reward, done, truncated, self.info
    
    def reset(self, seed=None, options=None):
        global done
        global gotRewardObs
        gotRewardObs = False
        done = False
        
        self.info = {}
        return obs, self.info
    

# Class that handles training the model in a separate thread
class trainModel(threading.Thread):
    def __init__(self, env, algo='PPO', timesteps=10000, it=3):
        super().__init__(target=self, daemon=True)
        self.env = env
        self.algo = algo.upper()
        self.timesteps = timesteps
        self.it = it
        self.models_dir = f"models/{self.algo}_{int(time.localtime().tm_mon)}-{int(time.localtime().tm_mday)}-{int(time.localtime().tm_year)}_{int(time.localtime().tm_hour-6)}.{int(time.localtime().tm_min)}.{int(time.localtime().tm_sec)}"
        log_dir = f"logs/{self.algo}_{int(time.localtime().tm_mon)}-{int(time.localtime().tm_mday)}-{int(time.localtime().tm_year)}_{int(time.localtime().tm_hour-6)}.{int(time.localtime().tm_min)}.{int(time.localtime().tm_sec)}"

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        if (self.algo == "PPO"): 
            self.model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
        elif (self.algo == "A2C"):
            self.model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
        
        self.start()

    def run(self):
        for i in range(0,self.it):
            self.model.learn(total_timesteps=self.timesteps, reset_num_timesteps=False, tb_log_name=self.algo)
            self.model.save(f"{self.models_dir}/{self.timesteps*(i+1)}")
        # self.join()

# Step 1: define the handlers for your server.
#         I.e. When an object is sent with a method name, create a function to be run when that message is received.

async def say(m):
    print(m['params']['text'])

# Starts training the model
async def start(m):
    vals = m['params']['init']
    global obs
    obs = np.ones(vals[3])
    env = CustomEnv(vals[0], vals[1], vals[2], vals[3], vals[4])
    env.reset()
    global started
    started = True
    global myThread
    myThread = trainModel(env=env)

# Update global variables with state data from the client
async def stateRewObs(m):
    global reward
    global obs
    global done
    global gotRewardObs
    reward = m['params']['reward']
    obs = np.array(m['params']['obs'])
    done = m['params']['done']
    gotRewardObs = True

handlers = {
    "say": say,
    "start": start,
    "stateRewObs": stateRewObs
}

# Step 2: create a new server.
#         Include the value of the port on which you want to listen, as well as the handlers object you just created.
#         Beyond this, the server is running.
myNewServer = ws_server(3000, handlers)
myNewServer.set_list_mode(True)
myNewServer.set_broadcastable(True)
myNewServer.set_debug_mode(False)
myNewServer.set_probe_mode(True, 3000)

# Sends action and wait for state data from client
async def getState(action):
    await myNewServer.broadcast_message("step", {"action": action})
    while not gotRewardObs:
        continue

# Step 3: write code to interact with client(s).
# Loops asyncio.sleep() to keep the event loop (which contains the server) from finishing
async def broadcast_messages():
    await asyncio.sleep(5)
    while True:
        await asyncio.sleep(10)

async def main():
    server_task = asyncio.create_task(myNewServer.start_server())
    broadcast_task = asyncio.create_task(broadcast_messages())
    await asyncio.gather(server_task, broadcast_task)

if __name__ == "__main__":
    asyncio.run(main())
