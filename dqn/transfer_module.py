import numpy as np
import torch
from configuration import DEVICE
from dqn.replay_memory import ReplayMemory

class TransferModule:
    def __init__(self, models, loss):
        self.loss = loss
        self.models = models
        self.reset()

    def reset(self):
        # self.errors = np.zeros(len(self.models))
        # self.sum_errors = np.zeros(len(self.models))
        # self.i_updates = 0
        self.memory = ReplayMemory(1000)

    def push(self,*args):
        self.memory.push(*args)

    def _errors_with_tensors(self,x):
        losses = [self.loss(model(x), x).item() for model in self.models]
        return losses

    def errors(self):

        x =self.memory.sample_to_numpy(len(self.memory))
        x = torch.from_numpy(x).float().to(DEVICE)


        return self._errors_with_tensors(x)
        # print(losses)
        # errors = np.fromiter(losses, float, len(self.models))
        # self.sum_errors += errors
