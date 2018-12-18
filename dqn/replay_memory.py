from collections import namedtuple
import random

Transition = namedtuple("Transition", ("s", "a", "r_", "s_"))
Transition.__str__ = lambda self: "".join(
    ["(", str(self.s), ", ", str(self.a), ", ", str(self.r_), ", ", str(self.s_), ")"])


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __str__(self):
        rez = "".join(["{:05d} {} \n".format(it, str(t)) for it, t in enumerate(self.memory)])
        return rez[:-1]
