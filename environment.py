
from agent import Agent
from settings import WORLD_BOUNDS, NUM_AGENTS, SIMULATION_TICKS, NUM_FOOD, NUM_POISON, AGENT_SPEED, \
    AGENT_SIZE, FOOD_SIZE, POISON_SIZE, FOOD_REWARD, POISON_REWARD, CHECKPOINT_INTERVAL, AGENT_SIZE, AGENT_GRID_SIZE, SINGLE_GRID_SIZE
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
        self.agents = {}
        self.num_agents = None
        self.invalidate_agents = True

        self.food = []
        self.poison = []

        self.bounds = WORLD_BOUNDS # tuple e.g. (x, y)

        #self.pool = None if NUM_CORES < 2 else multiprocessing.Pool(NUM_CORES)
        self.generation = 0
        self.rewards = {}

        self.episodes = []
        self.episode_data = []
        self.episode_fitness = []

        self.init()

        # Determine path to configuration file. This path manipulation is
        # here so that the script will run successfully regardless of the
        # current working directory.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-feedforward')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_path)

        self.pop = neat.Population(self.config)
        self.stats = neat.StatisticsReporter()
        self.pop.add_reporter(self.stats)
        self.pop.add_reporter(neat.StdOutReporter(True))
        # Checkpoint every 25 generations or 900 seconds.
        self.pop.add_reporter(neat.Checkpointer(25, 900))

    def run(self):
        winner = self.pop.run(self.evaluate_genomes, 30)
        print(winner)

        #network_visualizer.plot_stats(self.stats, ylog=True, view=True, filename="feedforward-fitness.svg")
        #network_visualizer.plot_species(self.stats, view=True, filename="feedforward-speciation.svg")

        #node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
        network_visualizer.draw_net(self.config, winner, True)


        #network_visualizer.draw_net(self.config, winner, view=True,
        #                   filename="winner-feedforward.gv")
        #network_visualizer.draw_net(self.config, winner, view=True,
        #                   filename="winner-feedforward-enabled.gv", show_disabled=False)
        #network_visualizer.draw_net(self.config, winner, view=True,
        #                       filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)

    def get_scaled_inputs(self, agent):
        pos = agent.pos

        grid = []
        for idx in range(AGENT_GRID_SIZE*AGENT_GRID_SIZE):
            grid.append(0)

        grid_radius = int(AGENT_GRID_SIZE // 2) # get's the number of grid spaces on each side of the agent
        x0grid = int(pos[0] - ((AGENT_GRID_SIZE * grid_radius) + (AGENT_SIZE / 2)))
        y0grid = int(pos[1] - ((AGENT_GRID_SIZE * grid_radius) + (AGENT_SIZE / 2)))
        x1grid = int(x0grid + (AGENT_SIZE * AGENT_GRID_SIZE))
        y1grid = int(y0grid + (AGENT_SIZE * AGENT_GRID_SIZE))
        gridsize = (AGENT_SIZE * AGENT_GRID_SIZE)

        for idx in range(len(self.food)):
            xrpos = int(self.food[idx][0] - x0grid)
            yrpos = int(self.food[idx][1] - y0grid)
            if xrpos >= 0 and yrpos >= 0 and xrpos < gridsize and yrpos < gridsize:
                # A thing is in the grid
                xrpos = int(xrpos / AGENT_SIZE)
                yrpos = int(yrpos / AGENT_SIZE)
                index = xrpos + yrpos * AGENT_GRID_SIZE
                grid[index] = 1
                #print(index)

        for idx in range(len(self.poison)):
            xrpos = int(self.poison[idx][0] - x0grid)
            yrpos = int(self.poison[idx][1] - y0grid)
            if xrpos >= 0 and yrpos >= 0 and xrpos < gridsize and yrpos < gridsize:
                # A thing is in the grid
                xrpos = int(xrpos / AGENT_SIZE)
                yrpos = int(yrpos / AGENT_SIZE)
                index = xrpos + yrpos * AGENT_GRID_SIZE
                grid[index] = -1
                #print(index)

        return [int(i) for i in grid]

    def get_distance(self, pos1, pos2):
        delta_x = pos1[0] - pos2[0]
        delta_y = pos1[1] - pos2[1]

        return (math.sqrt(delta_x*delta_x + delta_y*delta_y))

    def step(self, actions, agent):

        reward = 0

        pos_x = agent.pos[0]
        pos_y = agent.pos[1]

        pos_x += actions[0] * AGENT_SPEED
        pos_y += actions[1] * AGENT_SPEED

        if pos_x > WORLD_BOUNDS[0] / 2:
            pos_x = -WORLD_BOUNDS[0] / 2
        if pos_x < -WORLD_BOUNDS[0] / 2:
            pos_x = WORLD_BOUNDS[0] / 2

        if pos_y > WORLD_BOUNDS[1] / 2:
            pos_y = -WORLD_BOUNDS[1] / 2
        if pos_y < -WORLD_BOUNDS[1] / 2:
            pos_y = WORLD_BOUNDS[1] / 2

        agent.set_pos((int(pos_x), int(pos_y)))

        toRemove_food = []
        for pos in self.food:
            if self.get_distance(agent.pos, pos) < FOOD_SIZE + AGENT_SIZE:
                reward += FOOD_REWARD
                toRemove_food.append(pos)

        toRemove_poison = []
        for pos in self.poison:
            if self.get_distance(agent.pos, pos) < POISON_SIZE + AGENT_SIZE:
                reward += POISON_REWARD
                toRemove_poison.append(pos)

        for elem in toRemove_food:
            self.food.remove(elem)

        for elem in toRemove_poison:
            self.poison.remove(elem)

        return (False, reward)

    def simulate(self, networks):

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

            for gid, genome, net in networks:
                if self.invalidate_agents:
                    self.init_agents(networks)
                    self.invalidate_agents = False

                if gid not in self.agents:
                    self.agents[gid] = Agent(gid)

                a = self.agents[gid]

                inputs = self.get_scaled_inputs(a)  # a.get_scaled_inputs(self)
                output = net.activate(inputs)

                dead, reward = self.step(output, a)
                genome.fitness += reward

                if gid not in self.rewards:
                    self.rewards[gid]= []
                self.rewards[gid].append(reward)

            if self.generation % CHECKPOINT_INTERVAL == 0:
                self.visualizer.update_view(self)
            self.respawn_items()

        if self.generation % CHECKPOINT_INTERVAL == 0:
            self.visualizer.flush()

        for k in self.rewards:
            scores.append(sum(self.rewards[k]))

        print("Score range [{:.3f}, {:.3f}]".format(min(scores), max(scores)))


    def evaluate_genomes(self, genomes, config):
        self.generation += 1

        t0 = time.time()
        forward_networks = []
        for gid, genome in genomes:
            forward_networks.append((gid, genome,
                    neat.nn.FeedForwardNetwork.create(genome, config)))
            genome.fitness = 0
        self.simulate(forward_networks)

        print("evaluated genome in {0}".format(time.time() - t0))

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

    def get_random_pos(self):
        x_bound = WORLD_BOUNDS[0] / 2
        rand_x = random.randrange(-x_bound, x_bound)
        y_bound = WORLD_BOUNDS[1] / 2
        rand_y = random.randrange(-y_bound, y_bound)
        return (rand_x, rand_y)

    def init_agents(self, networks):
        print("Placing agents...")
        self.agents.clear()
        self.num_agents = len(networks)
        for gid, genome, net in networks:
            if gid not in self.agents:
                self.agents[gid] = Agent(gid)
            a = self.agents[gid]
            a.set_pos(self.get_random_pos())

    def place_foods(self):
        self.food = []
        print("Placing foods...")
        for _ in range(NUM_FOOD):
            self.food.append(self.get_random_pos())

    def place_poisons(self):
        self.poison = []
        print("Placing poisons...")
        for _ in range(NUM_POISON):
            self.poison.append(self.get_random_pos())
