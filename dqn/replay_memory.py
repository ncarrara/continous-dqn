import random
from dqn.transition import Transition
import torch
import numpy as np

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_to_numpy(self,batch_size):
        transitions = self.sample(batch_size)
        batch = Transition(*zip(*transitions))
        s = np.array(batch.s)
        s_ = np.array(batch.s_)
        # TODO , if dim a > 1 , then dont use brackets
        a = np.transpose(np.array([batch.a]))
        r_ = np.transpose(np.array([batch.r_]))
        # print(s)
        # print(a)

        # print(s.shape)
        # print(s_.shape)
        # print(a.shape)
        # print(r_.shape)
        # print(s.shape)
        return np.concatenate((s, a, r_, s_),axis=1)


    def __len__(self):
        return len(self.memory)

    def __str__(self):
        rez = "".join(["{:05d} {} \n".format(it, str(t)) for it, t in enumerate(self.memory)])
        return rez[:-1]
