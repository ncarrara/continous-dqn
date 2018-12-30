import logging

from ncarrara.utils.configuration import Configuration
from ncarrara.utils.math import set_seed

logger = logging.getLogger(__name__)


class ConfigurationBFTQ_PYDIAL(Configuration):

    def load(self, path_config):
        super(ConfigurationBFTQ_PYDIAL,self).load(path_config)
        self.pydial_configuration = self.dict["general"]["pydial_configuration"]



C = ConfigurationBFTQ_PYDIAL().load_pytorch()
