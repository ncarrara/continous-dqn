import numpy as np
import torch
from continuous_dqn.tools.configuration import C
from utils_rl.transition.replay_memory import ReplayMemory
from utils_rl.transition.transition import TransitionGym


class TransferModule:
    def __init__(self, models, loss, experience_replays=None):
        self.loss = loss
        self.models = models
        self.experience_replays = experience_replays
        self.reset()

    def reset(self):
        self.memory = ReplayMemory(1000, TransitionGym)

    def push(self, *args):
        self.memory.push(*args)

    def _errors_with_tensors(self, x):
        losses = [self.loss(model(x), x).item() for model in self.models]
        return losses

    def errors(self):
        x = self.memory.sample_to_numpy(len(self.memory))
        x = torch.from_numpy(x).float().to(C.device)

        self._errors = self._errors_with_tensors(x)
        return self._errors

    def best_source_transitions(self):
        if self.experience_replays is None:
            raise Exception("must setup experience replays")
        errors = self.errors()

        i = np.argmin(errors)
        return self.experience_replays[i], errors
