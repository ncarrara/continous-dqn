import torch
import logging
import numpy as np
import json
import pprint
import os
from abc import ABC, abstractmethod

from ncarrara.utils.math import set_seed
from ncarrara.utils.os import makedirs
from ncarrara.utils.torch import set_device


class Configuration(object):

    def __init__(self, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.dict = {}
        self.device = None

    def __getitem__(self, arg):
        self.__check__()
        return self.dict[arg]

    def __str__(self):
        return pprint.pformat(self.dict)

    def __check__(self):
        if not self.dict:
            raise Exception("please load the configuration file")

    def create_fresh_workspace(self):
        self.__check__()
        self.__clean_workspace()
        makedirs(self.workspace)
        return self

    def __clean_workspace(self):
        self.__check__()
        os.system("rm -rf {}".format(self.workspace))
        return self

    def load(self, path_config):
        with open(path_config, 'r') as infile:
            self.dict = json.load(infile)

        self.id = self.dict["general"]["id"]
        self.workspace = self.dict["general"]["workspace"]

        # self.logging_level = self.dict["general"]["level"]
        # self.logging_format = self.dict["general"]["format"]
        self.seed = self.dict["general"]["seed"]
        # print(self.logging_format)
        # logging.basicConfig(
        #     # format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        #     level=logging.getLevelName(self.logging_level)
        # )
        import logging.config as config
        config.dictConfig(self.dict["general"]["dictConfig"])

        if self.seed is not None:
            set_seed(self.seed)

        if str(torch.__version__) == "0.4.1.":
            self.logger.warning("0.4.1. is bugged regarding mse loss")
        np.set_printoptions(precision=2)
        self.logger.info("Pytorch version : {}".format(torch.__version__))
        return self

    def load_matplotlib(self, graphic_engine):
        import matplotlib
        matplotlib.use(graphic_engine)
        return self

    def load_pytorch(self):
        _device = set_device()
        self.device = torch.device("cuda:" + str(_device) if torch.cuda.is_available() else "cpu")
        self.logger.info("DEVICE : ", self.device)
        return self
