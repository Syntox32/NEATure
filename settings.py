#!/usr/bin/python3

# Visualization and window parameters
WINDOW_TITLE = "NEATure"
RENDER_DEBUG = False
RESOLUTION = (500, 500)
AGENT_COLOR = (255, 255, 255, 0)
FOOD_COLOR = (0, 255, 0, 0)
POISON_COLOR = (255, 0, 0, 0)
GROUND_COLOR = (0, 0, 0, 0)
CHECKPOINT_INTERVAL = 5
DEBUG_RECT_COLOR = (255, 255, 255, 0)

# Environment
NUM_GENERATIONS = 10
NUM_AGENTS = 70
WORLD_BOUNDS = (1000, 1000)
FOOD_SIZE = 8
POISON_SIZE = 8
NUM_FOOD = 60
NUM_POISON = 30
LOOP_BOUNDS = False

# Evaluation
RUNS_PER_NETWORK = 5 # network runs for each evaluation
SIMULATION_TIME = 60 # seconds
TICKS_PER_SECOND = 20
SIMULATION_TICKS = SIMULATION_TIME * TICKS_PER_SECOND

# Agent Settings
AGENT_SIZE = 32  # diameter in pixels
AGENT_GRID_SIZE = 5  # makes a nxn grid around the agent, make sure n is an odd number!
SINGLE_GRID_SIZE = 24  # each grid is 16x16 pixels
AGENT_SPEED = 2

# Network
FOOD_REWARD = 200
POISON_REWARD = -400
AGENT_INTERACTION = -0.5
