import torch
from continuous_dqn.tools import utils
import logging
import json

from utils.configuration import Configuration

logger = logging.getLogger(__name__)

class ConfigurationContinuousDQN(Configuration):

    def load(self, path_config):
        super.load(path_config)
        self.path_sources = self.workspace + "/" + "sources"

        self.path_samples = self.path_sources + "/" + "samples"
        self.path_sources_params = self.path_sources + "/" + "params.json"
        self.path_models = self.path_sources + "/" + "models"

        self.path_targets = self.workspace + "/" + "targets"
        self.path_results_w_t = self.path_targets + "/" + "results_w_t.txt"
        self.path_results_wo_t = self.path_targets + "/" + "results_wo_t.txt"
        self.path_results_w_t_greedy = self.path_targets + "/" + "results_w_t_greedy.txt"
        self.path_results_wo_t_greedy = self.path_targets + "/" + "results_wo_t_greedy.txt"
        self.path_targets_params = self.path_targets + "/" + "params.json"

        self.makedirs(self.workspace)
        self.makedirs(self.path_sources)
        self.makedirs(self.path_samples)
        self.makedirs(self.path_models)
        self.makedirs(self.path_targets)


        if self.seed is not None:
            logger.info("SEED : {}".format(self.seed))
            utils.set_seed(self.seed)
        else:
            logger.info("NO SEED")

    def _load_params(self, path):
        if not self.dict:
            raise Exception("please load the configuration file first")
        with open(path, 'r') as file:
            params = json.load(file)
        logger.info(
            "[configuration] reading param from {} :\n{}".format(path, "".join([str(pa) + "\n" for pa in params])))
        return params

    def load_targets_params(self):
        return self._load_params(self.path_targets_params)

    def load_sources_params(self):
        return self._load_params(self.path_sources_params)


C = ConfigurationContinuousDQN()
