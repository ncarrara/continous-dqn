import torch
import logging
import numpy as np
import json
import pprint
import os

from ncarrara.utils.math import set_seed
from ncarrara.utils.torch import set_device




class Configuration(object):

    def __init__(self, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.dict = {}
        self.device = None

    def __getitem__(self, arg):
        if not self.dict:
            raise Exception("please load the configuration file")
        return self.dict[arg]

    def __str__(self):
        return pprint.pformat(self.dict)

    def load(self, path_config):
        with open(path_config, 'r') as infile:
            self.dict = json.load(infile)

        self.id = self.dict["general"]["id"]
        self.workspace = self.dict["general"]["workspace"]

        self.logging_level = self.dict["general"]["level"]
        self.seed = self.dict["general"]["seed"]

        logging.basicConfig(level=logging.getLevelName(self.logging_level))


        if self.seed is not None:
            set_seed(self.seed)

        if str(torch.__version__) == "0.4.1.":
            self.logger.warning("0.4.1. is bugged regarding mse loss")
        np.set_printoptions(precision=2)
        self.logger.info("Pytorch version : {}".format(torch.__version__))

    def load_matplotlib(self, graphic_engine):
        import matplotlib
        matplotlib.use(graphic_engine)  # template

    def load_pytorch(self):
        _device = set_device()
        self.device = torch.device("cuda:" + str(_device) if torch.cuda.is_available() else "cpu")
        self.logger.info("DEVICE : ", self.device)
        return self
