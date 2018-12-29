import logging

from utils.configuration import Configuration
from utils.math import set_seed

logger = logging.getLogger(__name__)


class ConfigurationBFTQ_PYDIAL(Configuration):

    def load(self, path_config):
        super.load(path_config)
        self.pydial_configuration = self.dict["general"]["pydial_configuration"]
        if self.seed is not None:
            logger.info("SEED : {}".format(self.seed))
            set_seed(self.seed)
        else:
            logger.info("NO SEED")


C = ConfigurationBFTQ_PYDIAL()
