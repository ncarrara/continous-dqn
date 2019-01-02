import logging

from ncarrara.utils.configuration import Configuration

logger = logging.getLogger(__name__)

# class ConfigurationBFTQ_PYDIAL(Configuration):

    # def load(self, path_config):
    #     super(ConfigurationBFTQ_PYDIAL,self).load(path_config)



C = Configuration().load_pytorch() #BFTQ_PYDIAL().load_pytorch()
