import random
import numpy as np
from dqn.replay_memory import ReplayMemory
import torch
import os
import logging

logger = logging.getLogger(__name__)


def epsilon_decay(start=1.0, decay=0.01, N=100):
    return [np.exp(-n / (1. / decay)) * start for n in range(N)]


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


def load_autoencoders(path_autoencoders):
    from configuration import DEVICE
    logger.info("reading autoencoders at {}".format(path_autoencoders))
    files_autoencoders = os.listdir(path_autoencoders)
    autoencoders = [None] * len(files_autoencoders)
    for file in files_autoencoders:
        i_autoencoder = int(file.split(".")[0])
        path_autoencoder = path_autoencoders + "/" + file
        autoencoders[i_autoencoder] = torch.load(path_autoencoder, map_location=DEVICE)
    return autoencoders


def load_experience_replays(path_data):
    from configuration import C
    logger.info("reading samples ...")
    files = os.listdir(path_data)
    print(files)
    files.remove(C.PARAMS_FILE)
    ers = [None] * len(files)
    for file in files:
        id_env = int(file.split(".")[0])
        path_file = path_data + "/" + file
        logger.info("reading {}".format(path_file))
        rm = ReplayMemory(10000)
        rm.load_memory(path_file)
        ers[id_env] = rm
    return ers


def read_samples(path_data):
    from configuration import DEVICE
    ers = load_experience_replays(path_data)
    all_transitions = [None] * len(ers)
    for id_env, rm in enumerate(ers):
        data = rm.sample_to_numpy(len(rm))
        all_transitions[id_env] = torch.from_numpy(data).float().to(DEVICE)
    return all_transitions


def array_to_cross_comparaison(tab):
    from configuration import C
    params = C.load_sample_params()
    print(params)
    keys = params[0].keys()
    print(keys)
    TODO ad line diff on params
    head = "-"*6*len(params)+"\n"
    for key in keys:
        xx = ""
        for param in params:
            xx += "{:5.2f} ".format(param[key])
        head += "{} <- {}\n".format(xx,key)
    toprint = head+"-"*6*len(params)+"\n"
    for ienv in range(len(tab)):
        toprint += format_errors(tab[ienv], ienv) + "\n"
    return toprint


def format_errors(errors, ienv):
    toprint = ""
    bold_index = np.argmin(errors)
    for ienv2 in range(len(errors)):
        if ienv2 == bold_index:
            toprint += Color.BOLD + "{:5.2f} ".format(errors[ienv2]) + Color.END
        else:
            if ienv2 == ienv:
                toprint += Color.PURPLE + "{:5.2f} ".format(errors[ienv2]) + Color.END
            else:
                toprint += "{:5.2f} ".format(errors[ienv2])
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
