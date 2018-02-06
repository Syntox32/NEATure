
from agent import Agent
from settings import *
import neat
import os
import time
import network_visualizer
from visualization import Visualizer
import random
import math


class Environment:

    #
    # Each environment should have a config and a Population
    # the environment should also do the evaluation of the genomes
    #
    # each genome is given in a list with an ID and the actual Genome
    # we will create an agent for each ID/Genome and use that to simulate
    #
    # each evaluation will do improve the network for X generations
    # we then simulate the each network and observe its actions,
    # each action will change the properties of the Agent class it was given,
    # in the simulation we observe the action and what score it got from it,
    # this is called an evaluation
    #
    # each new evaluation will be called an episode
    # in each episode we store the overall score/fitness
    # the agents will be visualized for Y seconds before doing another
    # evaluation of the genomes.
    #
    # the genome fitness in our case is for an example how much food it has eaten,
    # the fitness will drop extremely when the agent eats poison
    #
    # the fitness in the beginning of the experiment would probably be the ratio
    # between food and poison
    #   fitness = food / poison, or fitness = food / (poison / 2)
    #

    def __init__(self):
        self.visualizer = Visualizer()
        self.render = False

        self.food = []
        self.poison = []
        self.generation = 1

        self.bounds = WORLD_BOUNDS # tuple e.g. (x, y)
        self.agent = Agent()

        #self.pool = None if NUM_CORES < 2 else multiprocessing.Pool(NUM_CORES)

        self.init()

    def get_scaled_inputs(self, agent):
        pos = agent.pos

        grid = []
        debug_grid_pos = []

        grid_radius = int(AGENT_GRID_SIZE // 2) # get's the number of grid spaces on each side of the agent
        x0grid = int(pos[0] - ((SINGLE_GRID_SIZE * grid_radius) + (SINGLE_GRID_SIZE / 2)))
        y0grid = int(pos[1] - ((SINGLE_GRID_SIZE * grid_radius) + (SINGLE_GRID_SIZE / 2)))
        #x1grid = int(x0grid + (AGENT_SIZE * AGENT_GRID_SIZE))
        #y1grid = int(y0grid + (AGENT_SIZE * AGENT_GRID_SIZE))
        gridsize = (SINGLE_GRID_SIZE * AGENT_GRID_SIZE)



        for y in range(AGENT_GRID_SIZE):
            for x in range(AGENT_GRID_SIZE):
                debug_grid_pos.append((x0grid + x * SINGLE_GRID_SIZE, y0grid + y * SINGLE_GRID_SIZE))
                grid.append(0)


        for idx in range(len(self.food)):
            xrpos = int(self.food[idx][0] - x0grid)
            yrpos = int(self.food[idx][1] - y0grid)

            if xrpos >= 0 and yrpos >= 0 and xrpos < gridsize and yrpos < gridsize:
                # A thing is in the grid

                xrpos = int(xrpos / SINGLE_GRID_SIZE)
                yrpos = int(yrpos / SINGLE_GRID_SIZE)
                index = xrpos + yrpos * AGENT_GRID_SIZE

                grid[index] = 1
                #print(index)

        for idx in range(len(self.poison)):
            xrpos = int(self.poison[idx][0] - x0grid)
            yrpos = int(self.poison[idx][1] - y0grid)
            if xrpos >= 0 and yrpos >= 0 and xrpos < gridsize and yrpos < gridsize:
                # A thing is in the grid
                xrpos = int(xrpos / SINGLE_GRID_SIZE)
                yrpos = int(yrpos / SINGLE_GRID_SIZE)
                index = xrpos + yrpos * AGENT_GRID_SIZE

                grid[index] = -1
                #print(index)

        if RENDER_DEBUG and self.render:
            self.visualizer.drawDebug(debug_grid_pos, grid)

        grid.append(agent.pos[0] / WORLD_BOUNDS[0])
        grid.append(agent.pos[1] / WORLD_BOUNDS[1])


        return [int(i) for i in grid]

    def get_distance(self, pos1, pos2):
        delta_x = pos1[0] - pos2[0]
        delta_y = pos1[1] - pos2[1]

        return (math.sqrt(delta_x*delta_x + delta_y*delta_y))

    def step(self, actions, agent):

        reward = 0

        pos_x = agent.pos[0]
        pos_y = agent.pos[1]

        if actions[0]:
            pos_x += AGENT_SPEED * actions[0]
        if actions[1]:
            pos_x -= AGENT_SPEED * actions[1]
        if actions[2]:
            pos_y += AGENT_SPEED * actions[2]
        if actions[3]:
            pos_y -= AGENT_SPEED * actions[3]


        if LOOP_BOUNDS:
            if pos_x > WORLD_BOUNDS[0]:
                pos_x = 0
            if pos_x < 0:
                pos_x = WORLD_BOUNDS[0]

            if pos_y > WORLD_BOUNDS[1]:
                pos_y = 0
            if pos_y < 0:
                pos_y = WORLD_BOUNDS[1]
        else:
            if pos_x > WORLD_BOUNDS[0]:
                pos_x = WORLD_BOUNDS[0]
            if pos_x < 0:
                pos_x = 0

            if pos_y > WORLD_BOUNDS[1]:
                pos_y = WORLD_BOUNDS[1]
            if pos_y < 0:
                pos_y = 0

        agent.set_pos((int(pos_x), int(pos_y)))

        toRemove_food = []


        for pos in self.food:
            if self.get_distance(agent.pos, pos) < (FOOD_SIZE + AGENT_SIZE) / 2:
                reward += FOOD_REWARD
                toRemove_food.append(pos)


        dead = False
        toRemove_poison = []
        for pos in self.poison:
            if self.get_distance(agent.pos, pos) < (POISON_SIZE + AGENT_SIZE) / 2:
                dead = True
                reward += POISON_REWARD
                toRemove_poison.append(pos)

        for elem in toRemove_food:
            self.food.remove(elem)

        for elem in toRemove_poison:
            self.poison.remove(elem)

        return (dead, reward)

    def simulate(self, genome, net):

        self.invalidate_agents = True

        if self.generation % CHECKPOINT_INTERVAL == 0 and False:
            self.visualizer.start_recording(("Video/" + str(self.generation) + ".mp4"))

        scores = []
        score = 0
        steps = 0
        self.place_foods()
        self.place_poisons()

        self.agent.set_pos(self.get_random_pos(agent_spawn=True))

        while steps < SIMULATION_TICKS:
            steps += 1
            old_pos = self.agent.pos

            if not self.agent.alive:
                break

            if self.generation % CHECKPOINT_INTERVAL == 0:
                self.visualizer.clear_view()

            inputs = self.get_scaled_inputs(self.agent)  # a.get_scaled_inputs(self)
            output = net.activate(inputs)

            dead, reward = self.step(output, self.agent)

            dist_moved = self.get_distance(old_pos, self.agent.pos)

            if dead:
                self.agent.alive = False

            score += reward

            if self.generation % CHECKPOINT_INTERVAL == 0 and False:
                self.visualizer.update_view(self)
            #self.respawn_items()

        if self.generation % CHECKPOINT_INTERVAL == 0 and False:
            self.visualizer.flush()

        return score

    def simulate_individual(self, networks):

        self.invalidate_agents = True

        if self.generation % CHECKPOINT_INTERVAL == 0:
            self.visualizer.start_recording(("Video/" + str(self.generation) + ".mp4"))

        scores = []
        rewards = {}
        steps = 0
        self.place_foods()
        self.place_poisons()

        while steps < SIMULATION_TICKS:
            steps += 1

            if self.generation % CHECKPOINT_INTERVAL == 0:
                self.visualizer.clear_view()

            for gid, genome, net in networks:
                if self.invalidate_agents:
                    self.init_agents(networks)
                    self.invalidate_agents = False

                if gid not in self.agents:
                    self.agents[gid] = Agent(gid)

                a = self.agents[gid]

                if not a.alive:
                    continue

                inputs = self.get_scaled_inputs(a)  # a.get_scaled_inputs(self)
                output = net.activate(inputs)

                dead, reward = self.step(output, a)

                if dead:
                    a.alive = False
                    continue

                genome.fitness += reward

                if gid not in self.rewards:
                    self.rewards[gid]= []
                self.rewards[gid].append(reward)

            if self.generation % CHECKPOINT_INTERVAL == 0:
                self.visualizer.update_view(self)
            #self.respawn_items()

        if self.generation % CHECKPOINT_INTERVAL == 0:
            self.visualizer.flush()

        for k in self.rewards:
            scores.append(sum(self.rewards[k]))

        print("Score range [{:.3f}, {:.3f}]".format(min(scores), max(scores)))

    def simulate_best(self, network):

        self.visualizer.start_recording(("Video/" + str(self.generation) + " - best" + ".mp4"))
        self.render = True

        scores = []
        rewards = {}
        steps = 0
        self.place_foods()
        self.place_poisons()

        self.agent.set_pos(self.get_random_pos(agent_spawn=True))

        while steps < SIMULATION_TICKS:
            steps += 1

            self.visualizer.clear_view()

            inputs = self.get_scaled_inputs(self.agent)  # a.get_scaled_inputs(self)
            output = network.activate(inputs)

            dead, reward = self.step(output, self.agent)

            self.visualizer.update_view(self)

        self.visualizer.flush()

    def respawn_items(self):
        food_count = len(self.food)
        respawn_chance = (NUM_FOOD - food_count) / NUM_FOOD
        if random.random() < respawn_chance and food_count < NUM_FOOD:
            self.food.append(self.get_random_pos())

        poison_count = len(self.poison)
        respawn_chance = (NUM_POISON - poison_count) / NUM_POISON
        if random.random() < respawn_chance and poison_count < NUM_POISON:
            self.poison.append(self.get_random_pos())

    def init(self):
        self.place_foods()
        self.place_poisons()

    def get_random_pos(self, agent_spawn=False):

        x_bound = WORLD_BOUNDS[0]
        y_bound = WORLD_BOUNDS[1]

        if agent_spawn:
            return int(x_bound/2), int(y_bound/2)

        while True:
            rand_x = random.randrange(0, x_bound)
            rand_y = random.randrange(0, y_bound)

            if self.get_distance((self.bounds[0] / 2, self.bounds[1] / 2), (rand_x, rand_y)) < SPAWN_AREA_SIZE:
                continue

            for f in self.food:
                if self.get_distance((rand_x, rand_y), f) < MIN_SPAWN_DIST:
                    continue

            for p in self.poison:
                if self.get_distance((rand_x, rand_y), p) < MIN_SPAWN_DIST:
                    continue

            return rand_x, rand_y

    def place_foods(self):
        self.food = []
        for _ in range(NUM_FOOD):
            self.food.append(self.get_random_pos())

    def place_poisons(self):
        self.poison = []
        for _ in range(NUM_POISON):
            self.poison.append(self.get_random_pos())
