#!/usr/bin/python3
import environment, os
from settings import *
from environment import Environment
from time import time
from settings import *
import neat
import os
import network_visualizer


def evaluate_genome(genome, config):
    network = neat.nn.RecurrentNetwork.create(genome, config)
    genome.fitness = 0

    e = Environment()

    return e.simulate(genome, network)

    best_network = None
    best_fit = -100000


def run():
    print("\n")
    print ("*" * 20 + " RUNNING NEATURE PARALLEL" + "*" * 20)
    print("\n\n\n\n")
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory./home/tuxinet
    print ("Loading configuration...")
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                              neat.DefaultSpeciesSet, neat.DefaultStagnation,
                              config_path)

    print("Generating population...")
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    pop.add_reporter(neat.Checkpointer(25, 900))

    print ("Engaging parallel evaluator...")
    parallel_evaluator = neat.ParallelEvaluator(2, evaluate_genome)

    winner = pop.run(parallel_evaluator.evaluate, NUM_GENERATIONS)
    print(winner)

    e = Environment()

    e.simulate_best(neat.nn.RecurrentNetwork.create(winner, config))

    #network_visualizer.plot_stats(self.stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    #network_visualizer.plot_species(self.stats, view=True, filename="feedforward-speciation.svg")

    #node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    network_visualizer.draw_net(config, winner, True)
    network_visualizer.plot_stats(stats, view=True)
    network_visualizer.plot_species(stats, view=True)

    #network_visualizer.draw_net(self.config, winner, view=True,
    #                   filename="winner-feedforward.gv")
    #network_visualizer.draw_net(self.config, winner, view=True,
    #                   filename="winner-feedforward-enabled.gv", show_disabled=False)
    #network_visualizer.draw_net(self.config, winner, view=True,
    #                       filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)

if __name__ == '__main__':

    local_dir = os.path.dirname(__file__)
    video_dir = os.path.join(local_dir, 'Video')
    if not os.path.isdir(video_dir):
        print("Could not find 'Video' directory, creating one...")
        os.makedirs(video_dir)

    run()

locals() # why is this line here again? // I dunno...
