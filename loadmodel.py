# Loads and runs existing models from the trainedExModels folder for testing, visualization, etc.

from stable_baselines3 import PPO
import os
import pygame
from stationaryGoalEnv import StationaryGoalEnv
from randomGoalEnv import RandomGoalEnv
from customEnv import CustomEnv
from game import Game
import tkinter as tk
from tkinter import filedialog

print("Enter environment goal type (stationary, random, custom): ")
while True:
    envType = input()
    if (envType.lower() == "stationary" or envType.lower() == "random" or envType.lower() == "custom"):
        break
    else:
        print("Invalid response. Please enter either 'stationary' 'custom' or 'random': ")

size = ['5','5']
if envType.lower() != "custom":
    print("Enter size in fromat 'x y' (leave blank for default): ")
    while True:
        sizeStr = input()
        if (sizeStr.count(' ') == 1 and sizeStr[0].isnumeric() and sizeStr[sizeStr.__len__()-1].isnumeric()):
            size = sizeStr.split(' ')
            break
        elif (not sizeStr):
            break
        else:
            print("Invalid response.")

print("Enter model path within the models folder: ")

root = tk.Tk()

filepath = filedialog.askopenfilename()
root.withdraw()

if not os.path.exists(f"{filepath}"):
    print("Must enter valid file path within trainedExModels folder")
    quit()

if (envType.lower() == "stationary"):
    env = StationaryGoalEnv(int(size[0]), int(size[1]))
elif (envType.lower() == "random"):
    env = RandomGoalEnv(int(size[0]), int(size[1]))
elif (envType.lower() == "custom"):
    env = CustomEnv(Game())

model = PPO("MlpPolicy", env, verbose=1)

# model.load(f"models/{str(filepath)}")
model.set_parameters(f"{str(filepath)}")
print(f"{filepath}")

episode = 0
while True:
    episode += 1
    obs, _ = env.reset()
    done = False
    score = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        # pygame.time.delay(50) # Uncomment this if you want to slow down simulation to see it better
    print('episode:{} score:{}'.format(episode, score))
    if (episode >= 500):
        episode = 0
        print("Enter 'y' to run again: ")
        if input() != 'y':
            break

env.close()