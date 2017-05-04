
from agent import Agent
from settings import WORLD_BOUNDS, NUM_AGENTS, SIMULATION_TICKS, NUM_FOOD, NUM_POISON
import neat
import os
import time
import network_visualizer
from visualization import Visualizer
import random

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
        self.num_agents = NUM_AGENTS

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
        winner = self.pop.run(self.evaluate_genomes, 5)
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
        return [0,1,1,0,-1,1,0,-1]

    def step(self, actions, agent):
        return (False, 200, "info")

    def simulate(self, networks):

        self.visualizer.start_recording("Video/out.mp4")

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
                #print(a.pos)

                inputs =  self.get_scaled_inputs(a)  # a.get_scaled_inputs(self)
                output = net.activate(inputs)
                #print(output)

                dead, reward, info = self.step(output, a)
                genome.fitness = reward

                if gid not in self.rewards:
                    self.rewards[gid]= []
                self.rewards[gid].append(reward)

            self.visualizer.update_view(self)
            self.respawn_items()

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
        self.simulate(forward_networks)

        self.visualizer.flush()

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
        self.num_agents = len(networks)
        for gid, genome, net in networks:
            print(gid)
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
        self.posion = []
        print("Placing poisons...")
        for _ in range(NUM_POISON):
            self.poison.append(self.get_random_pos())
