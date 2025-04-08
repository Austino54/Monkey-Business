Dependencies:
Python 3
Stable Baselines 3.0 - pip install stable-baselines3[extra]
PyGame - pip install pygame
Tensorboard - pip install tensorboard

To use tensorboard:
tensorboard --logdir=logs
Where "logs" is the filepath to the logs folder
IMPORTANT NOTE: Open settings in tensorboard and turn smoothing down. By default it is at .6 and may cause inaccurate graphs.

Usage:
Run modelTraining.py to train a model
Run loadmodel.py to load and run a model
Run playableMode.py to explore the stationary goal environment

Additional functionality coming soon