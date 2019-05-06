import numpy as np
import logging

from ncarrara.continuous_dqn.dqn.transfer_module import TransferModule
from ncarrara.utils.torch_utils import loss_fonction_factory
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.utils_rl.transition.transition import TransitionGym

logger = logging.getLogger(__name__)

import abc


class ActionTransferModule(TransferModule):
    def __init__(self,
                 loss_function,
                 gamma,
                 feature=None,
                 sources_params=None,
                 evaluate_continuously=False,
                 selection_method="best_fit",
                 Q_sources=None,
                 Q_partial_net=None,
                 Q_full_net=None,
                 device=None,
                 **kwargs):
        super().__init__(sources_params=sources_params, evaluate_continuously=evaluate_continuously,
                         selection_method=selection_method)
        self.memory = []
        self.Q_partial_net = Q_partial_net
        self.Q_full_net = Q_full_net
        self.Q_sources = Q_sources
        self.feature = feature
        self.device = device
        self.gamma = gamma
        self.loss_function = loss_fonction_factory(loss_function)
        self.actions = [[] for _ in range(len(Q_sources))]
        self.error_partial_net = np.random.rand(1)
        self.idx_last_best_net = None
        self.idx_current_best_net = None

    def _reset(self):
        self.memory.clear()
        self.error_partial_net = np.random.rand(1)

    def set_Q_full_net(self, net):
        self.Q_full_net = net

    def set_Q_partial_net(self, net):
        self.Q_partial_net = net

    def _update_best_net(self):
        idx_min_source_error = np.argmin(self.errors)
        if self.errors[idx_min_source_error] < self.error_partial_net:
            self.idx_current_best_net = idx_min_source_error
        else:
            self.idx_current_best_net = -1
        if self.idx_current_best_net != self.idx_last_best_net:

            if self.idx_current_best_net != -1:
                message = "switching to -> {}".format(self.sources_params[self.idx_current_best_net]["proba_hangup"])
            else:
                message = "switching to full net"
            logger.info(message)
        best_net = self.Q_full_net if self.idx_current_best_net == -1 else self.Q_sources[self.idx_current_best_net]
        self.idx_last_best_net = self.idx_current_best_net
        return best_net


    def _push_sample_to_memory(self, s, a, r_, s_, done, info):
        self.memory.append((s, a, r_, s_, done, info))

    def _compute_errors(self):
        last_transitions = self.memory[self.evaluation_index: self._memory_size()]
        all_transitions = TransitionGym(*zip(*self.memory))
        import torch
        with torch.no_grad():
            batch_all_transitions = TransitionGym(*zip(*all_transitions))
            state_batch = torch.cat(batch_all_transitions.s)
            actions_full_net = self.Q_full_net(state_batch).max(...)  # TODO

            batch = TransitionGym(*zip(*last_transitions))
            state_batch = torch.cat(batch.s)

            errors = []
            for idx, Q_source in enumerate(self.Q_sources):
                actions_Q_source = Q_source(state_batch).max(...)  # TODO
                self.actions[idx].extend(actions_Q_source)
                loss = actions_full_net != self.actions[idx]  # TODO
                errors.append(loss)
        errors = np.array(errors)
        errors = errors
        return errors / self._memory_size()

    def _memory_size(self):
        return len(self.memory)


