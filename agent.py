#!/usr/bin/python3

class Agent(object):
    def __init__(self, gid=-1):
        self.genome_id = gid
        self.pos = (0, 0)

    def do_action(self, net_output):
        print(net_output)

    def set_pos(self, pos):
        self.pos = pos

    def get_pos(self, pos):
        return self.pos
