#!/usr/bin/python3

# This part of the program is responsible for the vizualisation of the NEAT simulation

import pygame
import settings
from agent import Agent

class Visualizer(object):
    def __init__(self):
        pygame.init()

        self.windowCtx = pygame.display.set_mode(settings.RESOLUTION, 0, 32)
        pygame.display.set_caption(settings.WINDOW_TITLE)


    def draw_agent(self, agent:Agent):
        pygame.draw.circle(self.windowCtx, settings.AGENT_COLOR, agent.pos, 30, 0)
        pygame.display.update()
