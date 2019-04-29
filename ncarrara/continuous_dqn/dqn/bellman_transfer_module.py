import numpy as np
import logging

from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.utils_rl.transition.transition import TransitionGym

logger = logging.getLogger(__name__)

import abc


class SimpleTransferModule(abc.ABC):
    def __init__(self,
                 feature=None,
                 sources_params=None,
                 evaluate_continuously=False,
                 selection_method="best_fit",
                 Q_sources=None,
                 device=None,
                 **kwargs):
        super.__init__(sources_params, evaluate_continuously, selection_method)
        self.memory = []
        self.Q_sources = Q_sources
        self.feature = feature
        self.device=device

    def get_Q_source(self):
        return self.Q_sources[self.idx_best_fit]

    def _push_sample_to_memory(self, s, a, r_, s_, done, info):
        self.memory.append((s, a, r_, s_, done, info))

    def _compute_sum_last_errors(self):
        transitions = self.memory[self.evaluation_index: self._memory_size(self)]
        import torch
        with torch.no_grad():

            if type(transitions[0]) == type(torch.zeros(0)):
                transitions = torch.stack(transitions)
            else:
                transitions = torch.tensor(transitions).to(self.device)

            batch = TransitionGym(*zip(*transitions))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)),
                                          device=self.device,
                                          dtype=torch.uint8)
            non_final_next_states = [s for s in batch.s_ if s is not None]

            state_batch = torch.cat(batch.s)
            action_batch = torch.cat(batch.a)
            reward_batch = torch.cat(batch.r_)

            action_batch = action_batch.unsqueeze(1)

            next_state_values = torch.zeros(len(transitions), device=self.device)

            losses = []
            for Q_source in self.Q_sources:
                Q = Q_source(state_batch)
                state_action_values = Q.gather(1, action_batch)
                if non_final_next_states:
                    next_state_values[non_final_mask] = Q_source(torch.cat(non_final_next_states)).max(1)[0].detach()
                else:
                    logger.warning("Pas d'état non terminaux")
                self.expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
                loss = self.loss_function(state_action_values, self.expected_state_action_values.unsqueeze(1))
                losses.append(loss.item())
        losses = np.arrays(losses)
        return losses * len(transitions)

    def _memory_size(self):
        return len(self.memory)

    def _reset(self):
        self.memory.clear()

    def evaluate(self):
        """
        Evaluate the last unevaluated transitions
        :return:
        """
        if self.selection_method == "best_fit":
            sum_errors = self.sum_errors + self._compute_sum_last_errors()
        elif self.selection_method == "random":
            sum_errors = np.random.rand(self.N_sources)
        else:
            raise Exception("unkown selection methode : {}".format(self.selection_method))

        self._update_sum_errors(sum_errors)

        self.evaluation_index = self._memory_size() - 1
