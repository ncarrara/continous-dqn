import torch
import logging
import numpy as np
import json
import pprint
import os
import random
import torch
import numpy as np
import subprocess

def set_seed(seed):
    if seed is not None:
        logger.info("Setting seed = {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    gpu_memory = [int(x) for x in result.split()]
    return gpu_memory


def set_device():
    memory_map = get_gpu_memory_map()
    device = 0
    min = np.inf
    for k, v in enumerate(memory_map):
        logger.info("device={} memory used={}".format(k, v))
        # print type(v)
        if v < min:
            device = k
            min = v

    logger.info("setting process in device {}".format(device))
    torch.cuda.set_device(device)
    return device

logger = logging.getLogger(__name__)


class Configuration(object):

    def __init__(self, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self.dict = {}

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
        self.makedirs(self.workspace)
        level = self.dict["general"]["level"]
        self.seed = self.dict["general"]["seed"]
        self.pydial_configuration = self.dict["general"]["pydial_configuration"]
        if self.seed is not None:
            logger.info("SEED : {}".format(self.seed))
            set_seed(self.seed)
        else:
            logger.info("NO SEED")
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




_device = set_device()
DEVICE = torch.device("cuda:" + str(_device) if torch.cuda.is_available() else "cpu")
logger.info("DEVICE : ", DEVICE)
C = Configuration()
