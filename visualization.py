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

        for key, agent in env.agents.items():
            self.draw_agent(agent.pos)

        for food in env.food:
            self.draw_food(food)

        for poison in env.poison:
            self.draw_poison(poison)

        if self.video != None:
            self.video.write_frame(pygame.surfarray.pixels2d(self.windowCtx))

    def start_recording(self, filename="Video/out.mp4"):
        self.video = FFMPEG_VideoWriter(filename, settings.RESOLUTION, 20, withmask=True)

    def flush(self):
        self.video.close()
        self.video = None

    # Changing render position so that (0, 0) is in the center of the frame
    def transform_position(self, pos):
        render_pos_x = pos[0]
        render_pos_y = pos[1]

        render_pos_x += int(settings.RESOLUTION[0] / 2)
        render_pos_y += int(settings.RESOLUTION[1] / 2)

        return render_pos_x, render_pos_y

    def draw_agent(self, pos):
        pygame.draw.circle(self.windowCtx, settings.AGENT_COLOR, self.transform_position(pos), settings.AGENT_SIZE, 0)

    def draw_food(self, pos):
        pygame.draw.circle(self.windowCtx, settings.FOOD_COLOR, self.transform_position(pos), settings.FOOD_SIZE, 0)

    def draw_poison(self, pos):
        pygame.draw.circle(self.windowCtx, settings.POISON_COLOR, self.transform_position(pos), settings.POISON_SIZE, 0)
