Dependencies:
Python 3
Stable Baselines 3.0 - pip install stable-baselines3[extra]
PyGame - pip install pygame
Tensorboard - pip install tensorboard
Websockets - pip install websockets

To use tensorboard:
tensorboard --logdir=logs
Where "logs" is the filepath to the logs folder
IMPORTANT NOTE: Open settings in tensorboard and turn smoothing down. By default it is at .6 and may cause inaccurate graphs.

Usage:
Run modelTraining.py to train a model
Run loadmodel.py to load and run a model
Run playableMode.py to explore the stationary goal environment

For custom environments:
Use game.py as a template, make anything using pygame

For server use, first run Server/envServer.py, then run Client/envClient.py.
The client will run the game described in Client/playSpace.py and send data to the server, which handles the gym environment and model training

Additional functionality coming soon