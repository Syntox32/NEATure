#!/usr/bin/python3

class Agent(object):
    def __init__(self):
        self.pos = (0, 0)
        self.alive = True

    def do_action(self, net_output):
        print(net_output)

    def set_pos(self, new_pos):
        self.pos = new_pos

    def get_pos(self):
        return self.pos
