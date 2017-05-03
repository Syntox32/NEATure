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

    def update_view(self, env):

        self.windowCtx.Fill(settings.GROUND_COLOR)

        for agent in env.agents:
            self.draw_agent(agent.pos)

        for food in env.food:
            self.draw_food(food)

        for poison in env.poison:
            self.draw_poison(poison)

        pygame.display.update()


    def draw_agent(self, pos):
        pygame.draw.circle(self.windowCtx, settings.AGENT_COLOR, pos, 30, 0)

    def draw_food(self, pos):
        pygame.draw.circle(self.windowCtx, settings.FOOD_COLOR, pos, 3, 0)

    def draw_poison(self, pos):
        pygame.draw.circle(self.windowCtx, settings.POISON_COLOR, pos, 3, 0)
