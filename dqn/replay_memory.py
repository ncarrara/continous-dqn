import random
from dqn.transition import Transition
import torch

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

    def sample_to_tensors(self,batch_size):
        transitions = self.sample(batch_size)
        batch = Transition(*zip(*transitions))
        s = torch.Tensor(batch.s)
        s_ = torch.Tensor(batch.s_)
        # TODO , if dim a > 1 , then dont use brackets
        a = torch.t(torch.Tensor([batch.a]))
        r_ = torch.t(torch.Tensor([batch.r_]))
        return torch.cat((s,a,r_,s_),dim=1)

    def __len__(self):
        return len(self.memory)

    def __str__(self):
        rez = "".join(["{:05d} {} \n".format(it, str(t)) for it, t in enumerate(self.memory)])
        return rez[:-1]
