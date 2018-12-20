import random
import numpy as np
from dqn.replay_memory import ReplayMemory
import torch
import os
import logging

logger = logging.getLogger(__name__)


class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def read_samples(path_data="data/samples"):
    import configuration
    logger.info("reading samples ...")
    files = os.listdir(path_data)
    all_transitions = [None] * len(files)
    for file in files:
        id_env = int(file.split(".")[0])
        path_file = path_data + "/" + file
        logger.info("reading {}".format(path_file))
        rm = ReplayMemory(10000)
        rm.load_memory(path_file)
        data = rm.sample_to_numpy(len(rm))
        all_transitions[id_env] = torch.from_numpy(data).float().to(configuration.DEVICE)
    return all_transitions


def array_to_cross_comparaison(tab):
    # print("\n")
    # print("".join(["{}\n".format("".join(["{:.2f} ".format(a) for a in aa])) for aa in rez]))
    toprint = ""
    for ienv in range(len(tab)):
        bold_index = np.argmin(tab[ienv])
        for ienv2 in range(len(tab)):
            if ienv2 == bold_index:
                toprint += Color.BOLD + "{:.2f} ".format(tab[ienv][ienv2]) + Color.END
            else:
                if ienv2 == ienv:
                    toprint += Color.PURPLE + "{:.2f} ".format(tab[ienv][ienv2]) + Color.END
                else:
                    toprint += "{:.2f} ".format(tab[ienv][ienv2])
        toprint += "\n"
    return toprint


def set_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env is not None:
        env.seed(seed)




import torch
import numpy as np
import subprocess


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
