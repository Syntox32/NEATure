#!/usr/bin/python3

# Visualization and window parameters
WINDOW_TITLE = "NEATure"
RESOLUTION = (1000, 1000)
AGENT_COLOR = (255, 255, 255, 0)
FOOD_COLOR = (0, 255, 0, 0)
POISON_COLOR = (255, 0, 0, 0)
GROUND_COLOR = (0, 0, 0, 0)
CHECKPOINT_INTERVAL = 3

# Environment
NUM_AGENTS = 70
WORLD_BOUNDS = (1000, 1000)
FOOD_SIZE = 8
POISON_SIZE = 8
NUM_FOOD = 70
NUM_POISON = 35

# Evaluation
RUNS_PER_NETWORK = 5 # network runs for each evaluation
SIMULATION_TIME = 60 # seconds
TICKS_PER_SECOND = 20
SIMULATION_TICKS = SIMULATION_TIME * TICKS_PER_SECOND

# Agent Settings
AGENT_SIZE = 16  # diameter in pixels
AGENT_GRID_SIZE = 11  # makes a 7x7 grid around the agent
SINGLE_GRID_SIZE = 16  # each grid is 16x16 pixels
AGENT_SPEED = 2

# Network
FOOD_REWARD = 200
POISON_REWARD = -400
AGENT_INTERACTION = -0.5
