# Trains models with RandomGoalEnv using PPO. Replace "model" variable with A2C to use A2C, and replace "env" variable with StationaryGoalEnv to use a stationary goal.

from stable_baselines3 import A2C
from stable_baselines3 import PPO
import os
import time
from stationaryGoalEnv import StationaryGoalEnv
from randomGoalEnv import RandomGoalEnv

print("Enter algorithm to train model (PPO, A2C): ")
while True:
    algo = input()
    if (algo.upper() == "PPO" or algo.upper() == "A2C"):
        break
    else:
        print("Invalid response. Please enter either 'PPO' or 'A2C': ")

print("Enter environment goal type (stationary, random): ")
while True:
    envType = input()
    if (envType.lower() == "stationary" or envType.lower() == "random"):
        break
    else:
        print("Invalid response. Please enter either 'stationary' or 'random': ")

# run 'tensorboard --logdir=logs' to see training logs. Replace 'logs' with whatever filepath the logs folder has
models_dir = f"models/{algo.upper()}_{int(time.localtime().tm_mon)}-{int(time.localtime().tm_mday)}-{int(time.localtime().tm_year)}_{int(time.localtime().tm_hour-6)}.{int(time.localtime().tm_min)}.{int(time.localtime().tm_sec)}"
log_dir = f"logs/{algo.upper()}_{int(time.localtime().tm_mon)}-{int(time.localtime().tm_mday)}-{int(time.localtime().tm_year)}_{int(time.localtime().tm_hour-6)}.{int(time.localtime().tm_min)}.{int(time.localtime().tm_sec)}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if (envType.lower() == "stationary"):
    env = StationaryGoalEnv(10, 10)
elif (envType.lower() == "random"):
    env = RandomGoalEnv(10, 10)
env.reset()

if (algo.upper() == "PPO"): 
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
elif (algo.upper() == "A2C"):
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# for some reason model.load doesn't load the model data. Must use model.set_parameters instead to get the parameters
# model.load("models/PPO_2-18-2025_22.29.0/90000.zip")
# model.set_parameters("models/PPO_2-18-2025_22.29.0/90000.zip")

TIMESTEPS = 10000

for i in range(1,10):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=algo.upper())
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()