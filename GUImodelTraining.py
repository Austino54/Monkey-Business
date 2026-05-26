# Trains models

from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import time
from stationaryGoalEnv import StationaryGoalEnv
from randomGoalEnv import RandomGoalEnv
from customEnv import CustomEnv
# from GUIgame import Game
from snakeTest import Game


size = ['5','5']

# run 'tensorboard --logdir=logs' to see training logs. Replace 'logs' with whatever filepath the logs folder has
models_dir = f"models/{"PPO".upper()}_{int(time.localtime().tm_mon)}-{int(time.localtime().tm_mday)}-{int(time.localtime().tm_year)}_{int(time.localtime().tm_hour-6)}.{int(time.localtime().tm_min)}.{int(time.localtime().tm_sec)}"
log_dir = f"logs/{"PPO".upper()}_{int(time.localtime().tm_mon)}-{int(time.localtime().tm_mday)}-{int(time.localtime().tm_year)}_{int(time.localtime().tm_hour-6)}.{int(time.localtime().tm_min)}.{int(time.localtime().tm_sec)}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# if (envType.lower() == "stationary"):
#     env = StationaryGoalEnv(int(size[0]), int(size[1]))
# elif (envType.lower() == "random"):
#     env = RandomGoalEnv(int(size[0]), int(size[1]))
#     # env = DummyVecEnv([lambda: RandomGoalEnv(int(size[0]), int(size[1])), lambda: RandomGoalEnv(int(size[0]), int(size[1])), lambda: RandomGoalEnv(int(size[0]), int(size[1])), lambda: RandomGoalEnv(int(size[0]), int(size[1]))])
# elif (envType.lower() == "custom"):
env = CustomEnv(Game())
env.reset()

# if (algo.upper() == "PPO"): 
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
# elif (algo.upper() == "A2C"):
#     model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# for some reason model.load doesn't load the model data. Must use model.set_parameters instead to get the parameters
# model.load("models/PPO_2-18-2025_22.29.0/90000.zip")
# model.set_parameters("models/PPO_2-18-2025_22.29.0/90000.zip")

TIMESTEPS = 10000

for i in range(1,10):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO".upper())
    model.save(f"{models_dir}/{TIMESTEPS*(i+1)}")

env.close()