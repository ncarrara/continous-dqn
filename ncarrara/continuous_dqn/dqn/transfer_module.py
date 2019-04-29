import numpy as np
import logging

logger = logging.getLogger(__name__)

import abc


class TransferModule(abc.ABC):
    def __init__(self,
                 sources_params=None,
                 evaluate_continuously=False,
                 selection_method="best_fit", **kwargs):
        self.N_sources = len(sources_params)
        self.selection_method = selection_method  # best fit or random
        self.sources_params = sources_params
        self.evaluate_continuously = evaluate_continuously

    @abc.abstractmethod
    def _push_sample_to_memory(self, s, a, r_, s_, done, info):
        return

    @abc.abstractmethod
    def _compute_sum_last_errors(self):
        return

    @abc.abstractmethod
    def _memory_size(self):
        return

    @abc.abstractmethod
    def _reset(self):
        return


    def reset(self):
        self.idx_last_best_fit = None
        self._update_sum_errors(np.random.random_sample((1, self.N_sources))[0])
        self.evaluation_index = 0
        self._reset()

    def get_best_error(self):
        return self.errors[self.idx_best_fit]

    def push(self, s, a, r_, s_, done, info):
        self._push_sample_to_memory(s, a, r_, s_, done, info)
        if self.evaluate_continuously:
            self.update()

    def best_source_params(self):
        return  self.sources_params[self.idx_best_fit]

    def _update_sum_errors(self, sum_errors):
        self.sum_errors = sum_errors
        self.errors = self.sum_errors / (1 if self._memory_size() == 0 else self._memory_size())
        self.idx_best_fit = np.argmin(self.errors)
        if self.idx_last_best_fit is None or self.idx_best_fit != self.idx_last_best_fit:
            logger.info("Best fit changed [{}]: {}".format(self.idx_best_fit, self.sources_params[self.idx_best_fit]))
        self.idx_last_best_fit = self.idx_best_fit

    def push_memory(self, memory):
        for sample in memory:
            self._push_sample_to_memory(*sample)
        if self.evaluate_continuously:
            self.update()

    def update(self):
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


