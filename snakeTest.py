import os
import json
import random
import pygame
import sys
import numpy as np

WIDTH, HEIGHT = 600, 400
snakeSize = 20
ROW, COL = WIDTH // snakeSize, HEIGHT // snakeSize 
delay = 100
FPS = 60

# default config filename (trainGUI.py can write to this file)
DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "snake_reward_config.json")

class Game():
    actions = 4     # Number of actions the agent can take (currently discrete only)
    minObsVal = 0   # Minimum number that any observation value can be
    maxObsVal = 30   # Maximum number that any observation value can be
    numObs = 5      # Number of values in the observation space, i.e. how many values you will send the RL model
    obsType = np.int64    # Data type of the observations, from numpy

    def __init__(self, startingX=WIDTH // 2, startingY=HEIGHT // 2, size=snakeSize, color="green", config_path=None):
        pygame.init()
        self.win = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.apple = Food()
        self.size = size
        self.color = color
        self.length = 1
        self.x_vel = size
        self.y_vel = 0
        self.startingX = startingX
        self.startingY = startingY
        self.x = [startingX]
        self.y = [startingY]
        self.eaten = False
        self.running = True

        # config
        self.config_path = config_path or DEFAULT_CONFIG
        self.reward_items = self._load_reward_config(self.config_path)
        self._parse_reward_values()

    def step(self, action):
        # handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
        if action == -1:
            self.running = False
            return self.running

        # load latest reward config so GUI changes take effect during training
        self.reward_items = self._load_reward_config(self.config_path)
        self._parse_reward_values()

        # distance before move
        dx = abs(self.x[0] - self.apple.x)
        dy = abs(self.y[0] - self.apple.y)
        dist_before = (dx ** 2 + dy ** 2) ** 0.5

        if action == 0:
            self.up()
        elif action == 1:
            self.down()
        elif action == 2:
            self.left()
        elif action == 3:
            self.right()

        self.gameLogic()

        # distance after move
        curdx = abs(self.x[0] - self.apple.x)
        curdy = abs(self.y[0] - self.apple.y)
        dist_after = (curdx ** 2 + curdy ** 2) ** 0.5

        # clear screen then draw grid, snake and apple
        try:
            self.win.fill("white")
        except Exception:
            # if window not available, initialize display
            pygame.display.init()
            self.win = pygame.display.set_mode((WIDTH, HEIGHT))
            self.win.fill("white")

        for i in range(ROW):
            for j in range(COL):
                pygame.draw.rect(self.win, "black", (i * snakeSize, j * snakeSize, snakeSize, snakeSize), 1)

        self.draw(self.win)
        self.apple.draw(self.win)
        pygame.display.update()
        obs = np.array([self.x[0], self.y[0], self.apple.x, self.apple.y, self.length])

        reward = 0
        # apple eaten
        if self.eaten:
            reward += self.apple_pts
            self.eaten = False
        # distance change
        if dist_after < dist_before:
            reward += self.distance_pts
        else:
            reward += -abs(self.distance_pts)

        # failure
        if not self.running:
            reward += self.failure_pts

        return not self.running, reward, obs
    
    def reset(self):
        pygame.init()
        self.apple = Food()
        self.length = 1
        self.x_vel = snakeSize
        self.y_vel = 0
        self.x = [self.startingX]
        self.y = [self.startingY]
        self.eaten = False
        self.running = True
        # refresh reward config on reset
        self.reward_items = self._load_reward_config(self.config_path)
        self._parse_reward_values()
        return np.array([self.x[0], self.y[0], self.apple.x, self.apple.y, self.length])

    def eat(self):
        if self.x[0] == self.apple.x and self.y[0] == self.apple.y:
            self.eaten = True
            self.apple.randomize()
            self.grow()
            global delay
            if delay > 60:
                delay -= 5
    
    def grow(self):
        self.x.append(self.x[-1])
        self.y.append(self.y[-1])
        self.length += 1
    
    def collide(self):
        # when snake goes out of window it returns back
        if self.x[0] > WIDTH:
            self.x[0] = 0
        if self.x[0] < 0:
            self.x[0] = WIDTH
        if self.y[0] > HEIGHT:
            self.y[0] = 0
        if self.y[0] < 0:
            self.y[0] = HEIGHT
        # collide with it self
        for i in reversed(range(3, self.length)):
            if self.x[0] == self.x[i] and self.y[0] == self.y[i]:
                self.running = False
                # sys.exit(0)

    def draw(self, win):
        # draw head
        pygame.draw.rect(win, "magenta", (self.x[0], self.y[0], self.size, self.size))
        #  draw body
        for i in range(1, self.length):
            pygame.draw.rect(win, self.color, (self.x[i], self.y[i], self.size, self.size))
    
    def move(self):
        for i in reversed(range(1, self.length)):
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]
        
        self.x[0] = self.x[0] + self.x_vel
        self.y[0] = self.y[0] + self.y_vel

    def gameLogic(self):
        self.move()
        self.eat()
        self.collide()

    def up(self):
        if (self.x[0] < 0 or self.x[0] >= WIDTH) or (self.y[0] < 0 or self.y[0] >= HEIGHT):
            return 
        self.y_vel = -snakeSize
        self.x_vel = 0

    def down(self):
        if (self.x[0] < 0 or self.x[0] >= WIDTH) or (self.y[0] < 0 or self.y[0] >= HEIGHT):
            return 
        self.y_vel = snakeSize
        self.x_vel = 0
    
    def left(self):
        if (self.x[0] < 0 or self.x[0] >= WIDTH) or (self.y[0] < 0 or self.y[0] >= HEIGHT):
            return 
        self.x_vel = -snakeSize
        self.y_vel = 0

    def right(self):
        if (self.x[0] < 0 or self.x[0] >= WIDTH) or (self.y[0] < 0 or self.y[0] >= HEIGHT):
            return 
        self.x_vel = snakeSize
        self.y_vel = 0
        
    def quit(self):
        pygame.quit()

    def _load_reward_config(self, path):
        # Load reward config JSON produced/edited by trainGUI.py. If missing, create defaults.
        default = {
            "reward_items": [
                {"id": "DISTANCE_PTS", "label": "Distance", "description": "Points for getting closer to apple", "options": [-20, -10, 0, 5, 10, 15, 20], "value": 15},
                {"id": "APPLE_PTS", "label": "Apple Eaten", "description": "Points for eating apple", "options": [10, 20, 30, 40, 50, 100], "value": 50},
                {"id": "FAILURE_PTS", "label": "Failure", "description": "Points for failing", "options": [-500, -200, -100, -50, -10], "value": -100}
            ]
        }
        try:
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    json.dump(default, f, indent=2)
                return default['reward_items']
            with open(path, 'r') as f:
                data = json.load(f)
            if 'reward_items' in data:
                return data['reward_items']
            else:
                return default['reward_items']
        except Exception:
            return default['reward_items']

    def _parse_reward_values(self):
        # Set numeric attributes used by reward computation
        # Defaults if config missing keys
        self.distance_pts = 15
        self.apple_pts = 50
        self.failure_pts = -100
        try:
            for item in (self.reward_items or []):
                id = item.get('id', '')
                val = item.get('value', item.get('default', None))
                if val is None:
                    continue
                try:
                    val = int(val)
                except Exception:
                    try:
                        val = float(val)
                    except Exception:
                        continue
                if id == 'DISTANCE_PTS':
                    self.distance_pts = val
                elif id == 'APPLE_PTS':
                    self.apple_pts = val
                elif id == 'FAILURE_PTS':
                    self.failure_pts = val
        except Exception:
            pass

# end of snake class
        
# Food class
class Food:
    def __init__(self):
        self.x = random.randint(0, ROW-1) * snakeSize
        self.y = random.randint(0, COL-1) * snakeSize
    
    def randomize(self):
        self.x = random.randint(0, ROW-1) * snakeSize
        self.y = random.randint(0, COL-1) * snakeSize
        
    def draw(self, win):
        pygame.draw.rect(win, "red", (self.x, self.y, snakeSize, snakeSize))

# top level functions
def draw(win, snake, apple):
    # draw 2d grid
    for i in range(ROW):
        for j in range(COL):
            pygame.draw.rect(win, "black", (i * snakeSize, j * snakeSize, snakeSize, snakeSize), 1)

    snake.draw(win)
    apple.draw(win)

def update(clock):
    pygame.display.update()
    pygame.time.delay(delay)
    clock.tick(FPS)

def gameLogic(snake, apple):
    snake.move()
    snake.eat(apple)
    snake.collide()
    

# def main():
#     clock = pygame.time.Clock()
#     snake = Game(WIDTH // 2, HEIGHT // 2,snakeSize, "green")
#     apple = Food()
    
#     direction = "R"
#     run = True
#     while run:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 run = False
#                 break
#         win.fill("white") # clear screen

#         keys = pygame.key.get_pressed()
#         if keys[pygame.K_LEFT] and direction != "R":
#             direction = "L"
#             snake.left()
#         elif keys[pygame.K_RIGHT] and direction != "L":
#             direction = "R"
#             snake.right()
#         elif keys[pygame.K_UP] and direction != "D":
#             direction = "U"
#             snake.up()
#         elif keys[pygame.K_DOWN] and direction != "U":
#             direction = "D"
#             snake.down()
    
#         gameLogic(snake, apple)
#         draw(win, snake, apple)
#         update(clock)
    
#     pygame.quit()


# if __name__ == "__main__":
#     main()


