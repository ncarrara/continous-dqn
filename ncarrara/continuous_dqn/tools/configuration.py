from ncarrara.continuous_dqn.tools import utils
import logging
import json

from ncarrara.utils.configuration import Configuration
from ncarrara.utils.os import makedirs

logger = logging.getLogger(__name__)

class ConfigurationContinuousDQN(Configuration):

    def load(self, path_config):
        super(ConfigurationContinuousDQN,self).load(path_config)
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

        makedirs(self.workspace)
        makedirs(self.path_sources)
        makedirs(self.path_samples)
        makedirs(self.path_models)
        makedirs(self.path_targets)



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


C = ConfigurationContinuousDQN().load_pytorch()

