import os
import random
import numpy as np
import json

from utils_rl.transition.transition import TransitionGym


class ReplayMemory(object):

    def __init__(self, capacity,class_transtion=TransitionGym):
        self.capacity = capacity
        self.class_transtion=class_transtion
        self.memory = []
        self.position = 0

    def reset(self):
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.class_transtion(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_to_numpy(self, batch_size):
        transitions = self.sample(batch_size)
        batch = self.class_transtion(*zip(*transitions))
        s = np.array(batch.s)
        s_ = np.array(batch.s_)
        # TODO , if dim a > 1 , then dont use brackets
        a = np.transpose(np.array([batch.a]))
        r_ = np.transpose(np.array([batch.r_]))
        return np.concatenate((s, a, r_, s_), axis=1)

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
            reward = torch.tensor([reward], device=device)
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
