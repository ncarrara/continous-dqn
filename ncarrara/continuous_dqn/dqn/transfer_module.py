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
        self.best_net=None
        self.error=None
    @abc.abstractmethod
    def _push_sample_to_memory(self, s, a, r_, s_, done, info):
        return

    @abc.abstractmethod
    def _compute_errors(self):
        return

    @abc.abstractmethod
    def _memory_size(self):
        return

    @abc.abstractmethod
    def _reset(self):
        return
    def reset(self):
        self.evaluation_index = 0
        self.best_net = None
        self.error = None
        self._reset()

    @abc.abstractmethod
    def _update_best_net(self):
        return


    def push(self, s, a, r_, s_, done, info):
        self._push_sample_to_memory(s, a, r_, s_, done, info)
        if self.evaluate_continuously:
            self.update()


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
            self.errors = self._compute_errors()
        elif self.selection_method == "random":
            self.errors = np.random.rand(self.N_sources)
        else:
            raise Exception("unkown selection methode : {}".format(self.selection_method))


        self.evaluation_index = self._memory_size()
        self.best_net = self._update_best_net()


