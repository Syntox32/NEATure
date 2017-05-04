#!/usr/bin/python3

import visualization
import environment
import settings
from agent import Agent

import pygame


def run():
    clock = pygame.time.Clock()

    vis = visualization.Visualizer()

    a = Agent()

    sim_speed_realtime = True
    running = True
    while running:

        #if (sim_speed_realtime):
            #clock.tick(settings.TICKS_PER_SECOND)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        sim_speed_realtime = not sim_speed_realtime

    pygame.quit()



if __name__ == '__main__':
    #run()
    e = environment.Environment()
    e.run()
