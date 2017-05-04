#!/usr/bin/python3

# This part of the program is responsible for the vizualisation of the NEAT simulation

import pygame
import settings
from agent import Agent
from ffmpeg_writer import FFMPEG_VideoWriter

class Visualizer(object):
    def __init__(self):
        self.windowCtx = pygame.Surface(settings.RESOLUTION)
        self.video = None

    def update_view(self, env=None):

        if env is None:
            return

        self.windowCtx.fill(settings.GROUND_COLOR)

        for agent in env.agents:
            self.draw_agent(agent.pos)

        for food in env.food:
            self.draw_food(food)

        for poison in env.poison:
            self.draw_poison(poison)

        if self.video != None:
            self.video.write_frame(pygame.surfarray.pixels2d(self.windowCtx))

    def start_recording(self, filename="out.mp4"):
        self.video = FFMPEG_VideoWriter(filename, settings.RESOLUTION, 20, withmask=True)

    def flush(self):
        self.video.close()
        self.video = None

    def draw_agent(self, pos):
        pygame.draw.circle(self.windowCtx, settings.AGENT_COLOR, pos, settings.AGENT_SIZE, 0)

    def draw_food(self, pos):
        pygame.draw.circle(self.windowCtx, settings.FOOD_COLOR, pos, settings.FOOD_SIZE, 0)

    def draw_poison(self, pos):
        pygame.draw.circle(self.windowCtx, settings.POISON_COLOR, pos, settings.POISON_SIZE, 0)
