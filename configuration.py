import torch
import utils
import logging
import numpy as np
import json
import pprint
import os
import envs.envs_factory as ef

logger = logging.getLogger(__name__)


class Configuration(object):

    PARAMS_FILE = "params.json"

    def __init__(self, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self.dict = {}

    def __getitem__(self, arg):
        if not self.dict:
            raise Exception("please load the configuration file")
        return self.dict[arg]

    def __str__(self):
        return pprint.pformat(self.dict)

    def load(self, path_config):
        with open(path_config, 'r') as infile:
            self.dict = json.load(infile)

        self.id = path_config
        self.workspace = self.dict["general"]["workspace"]
        self.path_samples = self.dict["general"]["path_samples"]
        self.path_models = self.dict["general"]["path_models"]
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)
        level = self.dict["general"]["level"]
        self.seed = self.dict["general"]["seed"]
        utils.set_seed(self.seed)
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

    def load_sample_params(self):
        if not self.dict:
            raise Exception("please load the configuration file first")
        with open(self.workspace+"/"+self.path_samples+"/"+self.PARAMS_FILE,'r') as file:
            params = json.load(file)
        print(params)
        return params

device = utils.set_device()
DEVICE = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")
print("DEVICE : ", DEVICE)
C = Configuration()
