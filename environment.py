
from agent import Agent
from settings import WORLD_BOUNDS, NUM_AGENTS
import neat
import os

class Environment:

    def __init__(self):
        self.agents = []
        self.food = []
        self.poison = []

        self.bounds = WORLD_BOUNDS # tuple e.g. (x, y)
        self.num_agents = NUM_AGENTS

        self.init()

        # Determine path to configuration file. This path manipulation is
        # here so that the script will run successfully regardless of the
        # current working directory.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-feedforward')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_path)

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


    def init(self):
        self.init_agents()

        self.place_agents()
        self.place_foods()
        self.place_poisons()

    def init_agents(self):
        for _ in range(self.num_agents):
            a = Agent()
            self.agents.append(a)


    def place_agents(self):
        pass

    def place_foods(self):
        pass

    def place_poisons(self):
        pass
