# !!WARNING!! IS NOT YET IMPLEMENTED AND KINDA DOESNT WORK TOO GOOD
# This is a game that does NOT run with every step of model training.
# Instead, this game runs an independent simulation with the agent taking action only when step() is called for by the model
# This means that the speed of training and server data transfer will not affect the game speed

import pygame
import threading
import numpy as np
import time

EMPTY = 0
SAND = 1
WATER = 2

SANDMATS = [EMPTY, WATER]
WATERMATS = [EMPTY]

COLORS = [
    (0,0,0), # Empty
    (240,216,96), # Sand
    (20, 30, 255) # Water
]

class Game(threading.Thread):
    maxX = 300
    maxY = 300
    def __init__(self):
        super().__init__(target=self, daemon=True)

        pygame.init()
        self.screen = pygame.display.set_mode((self.maxX, self.maxY))

        self.pixels = np.zeros((self.maxX, self.maxY), dtype=np.int64)
        self.pixels[:, self.maxY-1] = 2
        self.pixels[20:250, 0] = 1
        self.pixels[50:200, 1] = 1
        self.pixels[100:150, 2] = 1

        self.drawScreen()

        self.start()

    def drawScreen(self):
        for idx, mat in np.ndenumerate(self.pixels):
                pygame.Surface.set_at(self.screen, idx, COLORS[mat])
        pygame.display.update()

    def pixelWeak(self, x, y, mats):
        for mat in mats:
            if self.pixels[x,y] == mat:
                return True
        return False

    def sandPhys(self, x, y):
        if y == self.maxY-1:
            return
        if self.pixelWeak(x, y+1, SANDMATS):
            self.pixels[x, y] = self.pixels[x, y+1]
            self.pixels[x, y+1] = SAND
        elif x < self.maxX-1 and self.pixelWeak(x+1, y+1, SANDMATS) and self.pixelWeak(x+1, y, SANDMATS):
            self.pixels[x+1, y] = self.pixels[x+1, y+1]
            self.pixels[x+1, y+1] = SAND
        elif x > 0 and self.pixelWeak(x-1, y+1, SANDMATS) and self.pixelWeak(x-1, y, SANDMATS):
            self.pixels[x-1, y] = self.pixels[x-1, y+1]
            self.pixels[x-1, y+1] = SAND

    def run(self):
        print("Started")
        # endTime = time.time() + 10
        while True:
            print("Here!")
            for event in pygame.event.get():
                pass

            tmpPix = np.array(self.pixels)
            for idx, mat in np.ndenumerate(tmpPix):
                if mat == SAND:
                    # print('Made it!!!!!')
                    self.sandPhys(idx[0], idx[1])
                    # print('Made it!')

            self.drawScreen()

            # pygame.display.update()
            print("There!")
            # pygame.time.delay(50)

game = Game()

while game.is_alive():
    pass