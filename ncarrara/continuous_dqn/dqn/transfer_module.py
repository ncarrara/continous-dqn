import numpy as np
import logging

logger = logging.getLogger(__name__)


class TransferModule:
    def __init__(self, auto_encoders, loss, feature, experience_replays=None, Q_sources=None,
                 evaluate_continuously=False,device=None):
        self.loss = loss
        self.device=device
        self.Q_sources = Q_sources
        self.auto_encoders = auto_encoders
        self.feature = feature
        self.evaluate_continuously = evaluate_continuously
        self.experience_replays = experience_replays
        if self.experience_replays is None:
            logger.warning("experience replays is None")
        if self.Q_sources is None:
            logger.warning("Q sources is None")
        self.memory = []
        self.reset()

    def reset(self):
        self.memory = []
        self.errors = np.zeros(len(self.auto_encoders))
        self.sum_errors = np.zeros(len(self.auto_encoders))
        self.best_fit = np.random.randint(0, len(self.auto_encoders))
        self.evaluation_index = 0

    def get_experience_replay_source(self):
        return self.experience_replays[self.best_fit]

    def get_Q_source(self):
        return self.Q_sources[self.best_fit]

    def get_error(self):
        return self.errors[self.best_fit]



    def push(self, s, a, r_, s_, done, info):
        vector = self.feature((s, a, r_, s_, done, info))
        self.memory.append(vector)
        if self.evaluate_continuously:
            import torch
            with torch.no_grad():
                vector = torch.tensor(vector)
                loss = np.array([self.loss(ae(vector), vector).item() for ae in self.auto_encoders]).cpu().item()
            self.sum_errors += loss
            self.errors = self.sum_errors / len(self.memory)
            self.best_fit = np.argmin(self.errors)
            self.evaluation_index += 1

    def push_memory(self,memory):
        for sample in memory:
            vector = self.feature(sample)
            self.memory.append(vector)
        if self.evaluate_continuously:
            self.evaluate()

    def evaluate(self):
        """
        Evaluate the last unevaluated transitions
        :return:
        """
        import torch
        with torch.no_grad():
            toevaluate = torch.tensor(self.memory[self.evaluation_index: -1]).to(self.device)
            # TODO maybe parrale compute of this
            losses = np.array([self.loss(ae(toevaluate), toevaluate).item() for ae in self.auto_encoders])
        self.sum_errors = self.sum_errors + losses * (len(self.memory) - self.evaluation_index)
        self.errors = self.sum_errors / len(self.memory)
        self.best_fit = np.argmin(self.errors)
        self.evaluation_index = len(self.memory) - 1

