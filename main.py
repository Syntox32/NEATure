#!/usr/bin/python3

import visualization
import environment
from agent import Agent

import pygame


def run():
    vis = visualization.Visualizer()

    a = Agent()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

pygame.quit()



if __name__ == '__main__':
    #run()
    e = environment.Environment()
    e.run()
