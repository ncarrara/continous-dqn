import numpy as np
import logging

logger = logging.getLogger(__name__)


class TransferModule:
    def __init__(self, autoencoders, loss_autoencoders, feature_autoencoders, experience_replays=None, Q_sources=None,
                 evaluate_continuously=False, device=None, selection_method="best_fit", **kwargs):
        self.selection_method = selection_method  # best fit or random
        self.loss = loss_autoencoders
        self.device = device
        self.Q_sources = Q_sources
        self.auto_encoders = autoencoders
        self.feature = feature_autoencoders
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

    def is_q_transfering(self):
        return self.Q_sources is not None

    def is_experience_replay_transfering(self):
        return self.experience_replays is not None

    def get_experience_replay_source(self):
        if self.experience_replays is not None:
            return self.experience_replays[self.best_fit]
        else:
            raise Exception("Transfer module's experience_replays are None")

    def get_Q_source(self):
        if self.Q_sources is not None:
            return self.Q_sources[self.best_fit]
        else:
            raise Exception("Transfer module's Q_sources are None")

    def get_error(self):
        import torch
        return torch.tensor([self.errors[self.best_fit]],device=self.device)

    def push(self, s, a, r_, s_, done, info):
        sample = (s, a, r_, s_, done, info)
        vector = self.feature(sample, self.device)
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

    def push_memory(self, memory):
        for sample in memory:
            vector = self.feature(sample, self.device)
            self.memory.append(vector)
        if self.evaluate_continuously:
            self.evaluate()

    def evaluate(self):
        """
        Evaluate the last unevaluated transitions
        :return:
        """
        if self.selection_method == "best_fit":
            import torch
            with torch.no_grad():
                # toevaluate = torch.stack(self.memory[self.evaluation_index: -1])
                toevaluate = self.memory[self.evaluation_index: len(self.memory)]
                if type(toevaluate[0]) == type(torch.zeros(0)):
                    toevaluate = torch.stack(toevaluate)
                else:
                    toevaluate = torch.tensor(toevaluate).to(self.device)
                # TODO maybe parrale compute of this
                losses = np.array([self.loss(ae(toevaluate), toevaluate).item() for ae in self.auto_encoders])

                self.sum_errors = self.sum_errors + losses * (len(self.memory) - self.evaluation_index)

        elif self.selection_method == "random":
            self.sum_errors = np.random.rand(len(self.auto_encoders))
        else:
            raise Exception("unkown selection methode : {}".format(self.selection_method))
        self.errors = self.sum_errors / len(self.memory)
        self.best_fit = np.argmin(self.errors)
        self.evaluation_index = len(self.memory) - 1
