from ncarrara.continuous_dqn.tools import utils
import logging
import json

from ncarrara.utils.configuration import Configuration
from ncarrara.utils.os import makedirs

logger = logging.getLogger(__name__)


class ConfigurationContinuousDQN(Configuration):

    def load(self, config):
        super(ConfigurationContinuousDQN, self).load(config)
        self.path_sources = self.workspace / "sources"
        self.path_targets = self.workspace / "targets"
        return self

    def _load_params(self, path):
        if not self.dict:
            raise Exception("please load the configuration file first")
        with open(path, 'r') as file:
            params = json.load(file)
        logger.info(
            "[configuration] reading param from {} :\n{}".format(path, "".join([str(pa) + "\n" for pa in params])))
        return params

    def load_targets_params(self):
        return self._load_params(self.path_targets / "params.json")

    def load_sources_params(self):
        return self._load_params(self.path_sources / "params.json")


C = ConfigurationContinuousDQN()
