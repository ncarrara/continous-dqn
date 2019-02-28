import numpy as np
import torch
import logging
from ncarrara.continuous_dqn.tools.configuration import C

logger = logging.getLogger(__name__)


class TransferModule:
    def __init__(self, auto_encoders, loss, feature, experience_replays=None, Q_source=None):
        self.loss = loss
        self.auto_encoders = auto_encoders
        self.feature = feature

        self.experience_replays = experience_replays
        self.Q_sources = Q_source
        self.error = None

        self.Q_source = None
        self.experience_replay_source = None

        if self.experience_replays is None and self.Q_sources is None:
            raise Exception("Experiences replays and Q sources are none. At least one of them must be filled")
        else:
            if self.experience_replays is None:
                logger.warning("experience replays is None")
            elif self.Q_sources is None:
                logger.warning("Q sources is None")
            else:
                # all good
                pass
        self.memory = []
        self.reset()

    def reset(self):
        self.memory = []

    def get_experience_replay_source(self):
        return self.experience_replay_source

    def get_Q_source(self):
        return self.Q_source

    def get_error(self):
        return self.error

    def push(self, s, a, r_, s_, done, info):
        vector = self.feature(s, a, r_, s_, done, info)
        self.memory.append(vector)

    def _errors_with_tensors(self, x):
        losses = [self.loss(ae(x), x).item() for ae in self.auto_encoders]
        return losses

    def evaluate_all_memory(self):
        x = np.array(self.memory)
        x = torch.from_numpy(x).float().to(C.device)

        self._errors = self._errors_with_tensors(x)
        return self._errors

    def best_source_transitions(self):
        errors = self.evaluate_all_memory()
        i = np.argmin(errors)
        experience_replay = None if self.experience_replays is None else self.experience_replays[i]
        q_source = None if self.Q_sources is None else self.Q_sources[i]

        return q_source, experience_replay, errors
