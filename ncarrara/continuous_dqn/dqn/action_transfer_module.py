import numpy as np
import logging

import torch

from ncarrara.continuous_dqn.dqn.transfer_module import TransferModule
from ncarrara.utils.color import Color
from ncarrara.utils.torch_utils import loss_fonction_factory
from ncarrara.utils_rl.transition.transition import TransitionGym

logger = logging.getLogger(__name__)


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
        self.actions = [torch.LongTensor([]).to(device) for _ in range(len(Q_sources))]
        self.error_partial_net = np.random.rand(1)
        self.idx_last_best_net = None
        self.idx_current_best_net = None

    def _reset(self):
        logger.info("Reseting Action Transfer Module")
        self.memory.clear()
        self.error_partial_net = np.random.rand(1)
        self.errors = np.random.rand(self.N_sources)
        self.best_net = self._update_best_net()

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
        if self.idx_last_best_net is None:
            logger.info(
                "initialising best net with {}".format(self.sources_params[self.idx_current_best_net]["proba_hangup"]))
        else:
            if self.idx_current_best_net != self.idx_last_best_net:

                if self.idx_current_best_net != -1:
                    message = "{}switching to -> {}{}".format(Color.BOLD,
                                                              self.sources_params[self.idx_current_best_net][
                                                                  "proba_hangup"], Color.END)
                else:
                    message = Color.BOLD + "switching to full net" + Color.END
                logger.info(message)
            else:
                pass
        best_net = self.Q_full_net if self.idx_current_best_net == -1 else self.Q_sources[self.idx_current_best_net]
        self.idx_last_best_net = self.idx_current_best_net
        return best_net

    def _push_sample_to_memory(self, s, a, r_, s_, done, info):
        self.memory.append((s, a, r_, s_, done, info))

    def _compute_errors(self):
        # import torch
        # logger.info("memory size : {}".format(self._memory_size()))
        # logger.info("self.evaluation_index : {}".format(self.evaluation_index))
        with torch.no_grad():
            # foward full net on the whole batch (since fullnet has been updated)
            batch_all_transitions = TransitionGym(*zip(*self.memory))
            state_batch_all_transitions = torch.cat(batch_all_transitions.s)
            # logger.info("forward {} transitions for full Q".format(len(self.memory)))
            a_fullnet = self.Q_full_net(state_batch_all_transitions).max(1)[1]

            # foward the remaining state
            last_transitions = self.memory[self.evaluation_index: self._memory_size()]
            # logger.info("forward {} transitions for each Q source".format(len(last_transitions)))
            batch = TransitionGym(*zip(*last_transitions))
            state_batch = torch.cat(batch.s)

            errors = np.zeros(len(self.Q_sources))
            for idx, Q_source in enumerate(self.Q_sources):
                actions_Q_source = Q_source(state_batch).max(1)[1]
                self.actions[idx] = torch.cat((actions_Q_source, self.actions[idx]))
                error = torch.sum(a_fullnet != self.actions[idx]).item() / len(self.actions[idx])
                errors[idx]=error
            # print(errors)

            a_partialnet = self.Q_partial_net(state_batch_all_transitions).max(1)[1]
            # print(a_fullnet)
            # print(a_partialnet)
            self.error_partial_net = torch.sum(a_fullnet != a_partialnet).item() / len(a_partialnet)
            print("partial net error : {:.2f} vs {}".format(self.error_partial_net,errors))
        return errors

    def _memory_size(self):
        return len(self.memory)
