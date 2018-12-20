import torch
import utils
import logging
import numpy as np
import json

import os

logger = logging.getLogger(__name__)


class Configuration():
    CONFIG = None

    def __init__(self):
        pass

    def load(self, path_config):
        with open(path_config, 'r') as infile:
            self.CONFIG = json.load(infile)
        workspace = self.CONFIG["general"]["workspace"]
        if not os.path.exists(workspace):
            os.makedirs(workspace)
        level = self.CONFIG["general"]["level"]
        seed = self.CONFIG["general"]["seed"]
        utils.set_seed(seed)
        if level == "INFO":
            logging.basicConfig(level=logging.INFO)
        elif level == "ERROR":
            logging.basicConfig(level=logging.ERROR)
        else:
            raise Exception("unknow logging level")

        if str(torch.__version__) == "0.4.1.":
            logger.warn("0.4.1. is bugged regarding mse loss")

        np.set_printoptions(precision=2)
        logger.info("Pytorch version : {}".format(torch.__version__))
        return self.CONFIG


device = utils.set_device()
DEVICE = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")
print("DEVICE : ",DEVICE)
C = Configuration()
