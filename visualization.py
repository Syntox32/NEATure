#!/usr/bin/python3

# This part of the program is responsible for the vizualisation of the NEAT simulation

import pygame
import settings

class Visualizer(object):
    def __init__(self):
        pygame.init()

        self.windowCtx = pygame.display.set_mode(settings.RESOLUTION, 0, 32)
        pygame.display.set_caption(settings.WINDOW_TITLE)
