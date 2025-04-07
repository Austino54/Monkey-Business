# Playable version of the stationary goal environment

import pygame
import time

GREY = (140,140,140)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)

# Types: GREY = empty, RED = end, BLUE = player, GREEN = finished
class Cell():
    def __init__(self, x, y, screen, color=GREY):
        self.x = x
        self.y = y
        self.color = color
        pygame.draw.circle(screen, self.color, (x*100+50, y*100+50), 25)

    def colorChange(self, screen, newColor):
        self.color = newColor
        pygame.draw.circle(screen, self.color, (self.x*100+50, self.y*100+50), 25)
    
class Player(Cell):
    def move(self, screen, target):
        self.colorChange(screen, GREY)
        self.x = target.x
        self.y = target.y
        # print((player.x,player.y))
        if target.color == RED:
            player.colorChange(screen, GREEN)
            return False
        player.colorChange(screen, BLUE)
        return True


pygame.init()
width = 5
height = 5
screen = pygame.display.set_mode((width*100, height*100))

grid = [[Cell(j, i, screen) for i in range(height)] for j in range(width)]
grid[0][4].colorChange(screen, RED)
player = Player(2, 2, screen, BLUE)

cooldown = 0
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
            break

    keys = pygame.key.get_pressed()
    if time.time() >= cooldown:
        cooldown = time.time() + 0.5
        if keys[pygame.K_LEFT] and player.x>0:
            running = player.move(screen, grid[player.x-1][player.y])
        elif keys[pygame.K_RIGHT] and player.x<width-1:
            running = player.move(screen, grid[player.x+1][player.y])
        elif keys[pygame.K_UP] and player.y>0:
            running = player.move(screen, grid[player.x][player.y-1])
        elif keys[pygame.K_DOWN] and player.y<height-1:
            running = player.move(screen, grid[player.x][player.y+1])
        else:
            cooldown = 0
        pygame.display.update()