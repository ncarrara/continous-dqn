import numpy as np
import torch

from ncarrara.continuous_dqn.tools.configuration import C


class TransferModule:
    def __init__(self, models, loss, experience_replays=None):
        self.loss = loss
        self.models = models
        self.experience_replays = experience_replays
        self.memory = []
        self.reset()

    def reset(self):
        self.memory = []

    def push(self, vector):
        self.memory.append(vector)

    def _errors_with_tensors(self, x):
        losses = [self.loss(model(x), x).item() for model in self.models]
        return losses

    def errors(self):
        x = np.array(self.memory)
        x = torch.from_numpy(x).float().to(C.device)

        self._errors = self._errors_with_tensors(x)
        return self._errors

    def best_source_transitions(self):
        if self.experience_replays is None:
            raise Exception("must setup experience replays")
        errors = self.errors()
        i = np.argmin(errors)
        return self.experience_replays[i], errors
