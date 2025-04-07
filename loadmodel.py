# Loads and runs existing models from the trainedExModels folder for testing, visualization, etc.

from stable_baselines3 import PPO
import os
import pygame
from stationaryGoalEnv import StationaryGoalEnv
from randomGoalEnv import RandomGoalEnv

print("Enter model path within the models folder: ")
filepath = input()

if not os.path.exists(f"trainedExModels/{filepath}"):
    print("Must enter valid file path within trainedExModels folder")
    pass

env = RandomGoalEnv(10,10)

model = PPO("MlpPolicy", env, verbose=1)

# model.load(f"models/{str(filepath)}")
model.set_parameters(f"trainedExModels/{str(filepath)}")
print(f"trainedExModels/{filepath}")

episode = 0
while True:
    episode += 1
    obs, _ = env.reset()
    done = False
    score = 0
    while not done:
        # env.render()
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        pygame.time.delay(100)
    print('episode:{} score:{}'.format(episode, score))
    if (episode >= 500):
        episode = 0
        print("Enter 'y' to run again: ")
        if input() != 'y':
            break

env.close()