import os
import random
import numpy as np
import json

from ncarrara.utils_rl.transition.transition import TransitionGym
import sys


class Memory(object):

    def __init__(self, capacity=sys.maxsize,class_transition=TransitionGym):
        self.capacity = capacity
        self.class_transition = class_transition
        self.memory = []
        self.position = 0

    def reset(self):
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.class_transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = len(self.memory) if batch_size > len(self.memory) else batch_size
        return random.sample(self.memory, batch_size)

    def save_memory(self, path, indent=0):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        memory = [t._asdict() for t in self.memory]
        if indent > 0:
            json_str = json.dumps(memory, indent=indent)
        else:
            json_str = json.dumps(memory)
        with open(path, 'w') as f:
            f.write(json_str)

    def load_memory(self, path):
        with open(path, 'r') as infile:
            memory = json.load(infile)
        self.reset()
        for idata, data in enumerate(memory):
            self.push(*data.values())

    def apply_feature_to_states(self, feature):
        for i in range(len(self.memory)):
            state, action, reward, next_state, done, info = self.memory[i]
            self.memory[i] = TransitionGym(feature(state), action, reward, feature(next_state), done, info)

    def to_tensors(self, device):
        import torch
        for i in range(len(self.memory)):

            state, action, reward, next_state, done, info = self.memory[i]
            state = torch.tensor([[state]], device=device, dtype=torch.float)
            if not done:
                next_state = torch.tensor([[next_state]], device=device, dtype=torch.float)
            else:
                next_state = None
            action = torch.tensor([[action]], device=device, dtype=torch.long)
            reward = torch.tensor([float(reward)], device=device)
            self.memory[i] = TransitionGym(state, action, reward, next_state, done, info)

        return self

    def __len__(self):
        return len(self.memory)

    def __str__(self):
        rez = "".join(["{:05d} {} \n".format(it, str(t)) for it, t in enumerate(self.memory)])
        return rez[:-1]

# m = ReplayMemory(1000)
# m.push([1, 2, 3], 4, 18, [5, 6, 7])
# m.push([1, 2, 3], 4, 18, [5, 6, 7])
# m.push([1, 2, 3], 4, 18, [5, 6, 7])
# m.dump_memory("memory.json")
# # import json
# # print json.dumps(m)
# m = ReplayMemory(100)
# m.load_memory("memory.json")
# print m.memory
