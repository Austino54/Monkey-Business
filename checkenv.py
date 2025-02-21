from stable_baselines3.common.env_checker import check_env
from stationaryGoalEnv import ReachGoalEnv

GREY = (140,140,140)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)

env = ReachGoalEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)