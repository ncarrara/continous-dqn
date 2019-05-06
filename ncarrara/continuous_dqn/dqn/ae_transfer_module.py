import numpy as np
import logging

from ncarrara.continuous_dqn.dqn.transfer_module import TransferModule

logger = logging.getLogger(__name__)



class AutoencoderTransferModule(TransferModule):
    def __init__(self,
                 feature_autoencoders,
                 loss_autoencoders,
                 sources_params=None,
                 experience_replays=None,
                 Q_sources=None,
                 evaluate_continuously=False,
                 device=None,
                 selection_method="best_fit",
                 N_actions=None, **kwargs):
        raise Exception("Decrepated")
        self.N_sources = len(Q_sources)
        self.N_actions = N_actions
        self.selection_method = selection_method  # best fit or random
        self.loss = loss_autoencoders
        self.sources_params = sources_params
        self.device = device
        self.Q_sources = Q_sources
        self.feature = feature_autoencoders
        self.evaluate_continuously = evaluate_continuously
        self.experience_replays = experience_replays
        if self.experience_replays is None:
            logger.warning("experience replays is None")
        if self.Q_sources is None:
            logger.warning("Q sources is None")
        self.memory_x = []
        self.memory_a = []
        self.reset()

    def _reset(self):
        self.memory_x = []
        self.memory_a = []
        self.last_best_fit = None
        self._update_sum_errors(np.random.random_sample((1, self.N_sources))[0])
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

    def get_best_error(self):
        import torch
        return torch.tensor(super().get_best_error(), device=self.device)


    def _push_sample_to_memory(self,s, a, r_, s_, done, info):
        x, y = self.feature((s, a, r_, s_, done, info), self.N_actions)
        self.memory_x.append(x)
        self.memory_a.append(y)


    def _compute_sum_last_errors(self):
        import torch
        with torch.no_grad():
            X = self.memory_x[self.evaluation_index: self._memory_size(self)]
            if type(X[0]) == type(torch.zeros(0)):
                X = torch.stack(X)
            else:
                X = torch.tensor(X).to(self.device)

            a = self.memory_a[self.evaluation_index: self._memory_size(self)]
            if type(a[0]) == type(torch.zeros(0)):
                a = torch.stack(a)
            else:
                a = torch.tensor(a).to(self.device)

            # il y a surement moyen de faire ca en mode matricielle
            losses = []
            for ae in self.auto_encoders:
                loss = self.loss(X, ae(X).gather(1, a)).item()
                losses.append(loss)
        losses = np.arrays(losses)
        return losses * len(X)

    def _memory_size(self):
        return len(self.memory_x)

