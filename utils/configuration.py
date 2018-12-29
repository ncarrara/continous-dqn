import torch
from continuous_dqn.tools import utils
import logging
import numpy as np
import json
import pprint
import os

logger = logging.getLogger(__name__)


class Configuration(object):

    def __init__(self, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self.dict = {}
        self.device = None

    def __getitem__(self, arg):
        if not self.dict:
            raise Exception("please load the configuration file")
        return self.dict[arg]

    def __str__(self):
        return pprint.pformat(self.dict)

    def makedirs(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def load(self, path_config):
        with open(path_config, 'r') as infile:
            self.dict = json.load(infile)

        self.id = self.dict["general"]["id"]
        self.workspace = self.dict["general"]["workspace"]

        level = self.dict["general"]["level"]
        self.seed = self.dict["general"]["seed"]

        if level == "INFO":
            logging.basicConfig(level=logging.INFO)
        elif level == "ERROR":
            logging.basicConfig(level=logging.ERROR)
        else:
            raise Exception("unknow logging level")

        if str(torch.__version__) == "0.4.1.":
            logger.warning("0.4.1. is bugged regarding mse loss")
        np.set_printoptions(precision=2)
        logger.info("Pytorch version : {}".format(torch.__version__))

    def load_matplotlib(self,graphic_engine):
        import matplotlib
        matplotlib.use(graphic_engine) # template


    def load_pytorch(self):
        _device = utils.set_device()
        self.device = torch.device("cuda:" + str(_device) if torch.cuda.is_available() else "cpu")
        logger.info("DEVICE : ", self.device)
        return self

