import logging
import json
from pathlib import Path

from ncarrara.utils.configuration import Configuration

logger = logging.getLogger(__name__)


class ConfigurationContinuousDQN(Configuration):

    def load(self, config):
        super(ConfigurationContinuousDQN, self).load(config)
        if self["general"]["path_sources"] is None:
            if "generate_sources" not in self:
                raise Exception("You must specify path_sources or generate_sources")
            self.path_sources = self.workspace / "sources"
        else:
            self.path_sources = Path(self["general"]["path_sources"])
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
