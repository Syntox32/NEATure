#!/usr/bin/python3

import visualization
from agent import Agent


def run():
    vis = visualization.Visualizer()

    a = Agent()

    while True:
        vis.draw_agent(a)


if __name__ == '__main__':
    run()
