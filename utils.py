import random
import numpy as np
import torch

def set_seed(seed,env=None):
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
        print("device={} memory used={}".format(k, v))
        # print type(v)
        if v < min:
            device = k
            min = v

    print("setting process in device {}".format(device))
    torch.cuda.set_device(device)