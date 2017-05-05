#!/usr/bin/python3

# This part of the program is responsible for the vizualisation of the NEAT simulation

import pygame
import settings
from agent import Agent
from ffmpeg_writer import FFMPEG_VideoWriter
import random

class Visualizer(object):
    def __init__(self):
        self.windowCtx = pygame.Surface(settings.RESOLUTION)
        self.video = None

        self.ratio_x = settings.WORLD_BOUNDS[0] / settings.RESOLUTION[0]
        self.ratio_y = settings.WORLD_BOUNDS[1] / settings.RESOLUTION[1]

    def clear_view(self):
        self.windowCtx.fill(settings.GROUND_COLOR)

    def update_view(self, env=None):

        if env is None:
            return

        for key, agent in env.agents.items():
            self.draw_agent(agent.pos)

        for food in env.food:
            self.draw_food(food)

        for poison in env.poison:
            self.draw_poison(poison)

        if self.video != None:
            self.video.write_frame(pygame.surfarray.pixels2d(self.windowCtx))

    def start_recording(self, filename="Video/out.mp4"):
        self.video = FFMPEG_VideoWriter(filename, settings.RESOLUTION, settings.TICKS_PER_SECOND, withmask=True)

    def flush(self):
        if self.video != None:
            self.video.close()
            self.video = None

    def drawDebug(self, gridTiles, content):

        size = int(settings.SINGLE_GRID_SIZE/self.ratio_x)

        for i in range(len(gridTiles)):

            pos = self.transform_position(gridTiles[i])

            if content[i] == 0:
                pygame.draw.rect(self.windowCtx, settings.DEBUG_RECT_COLOR, pygame.Rect(pos, (size, size)), 1)

            if content[i] == 1:
                pygame.draw.rect(self.windowCtx, settings.FOOD_COLOR, pygame.Rect(pos, (size, size)), 1)
            if content[i] == -1:
                pygame.draw.rect(self.windowCtx, settings.POISON_COLOR, pygame.Rect(pos, (size, size)), 1)




    # Changing render position so that (0, 0) is in the center of the frame
    def transform_position(self, pos):
        render_pos_x = pos[0] / self.ratio_x
        render_pos_y = pos[1] / self.ratio_y

        return int(render_pos_x), int(render_pos_y)

        render_pos_x += int(settings.RESOLUTION[0] / 2)
        render_pos_y += int(settings.RESOLUTION[1] / 2)

        return int(render_pos_x), int(render_pos_y)

    def draw_agent(self, pos):
        pygame.draw.circle(self.windowCtx, settings.AGENT_COLOR, self.transform_position(pos), int((settings.AGENT_SIZE / 2) / self.ratio_x), 0)

    def draw_food(self, pos):
        pygame.draw.circle(self.windowCtx, settings.FOOD_COLOR, self.transform_position(pos), int(settings.FOOD_SIZE / self.ratio_x), 0)

    def draw_poison(self, pos):
        pygame.draw.circle(self.windowCtx, settings.POISON_COLOR, self.transform_position(pos), int(settings.POISON_SIZE / self.ratio_x), 0)
