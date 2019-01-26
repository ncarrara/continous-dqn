import subprocess
import numpy as np

import logging
import os

logger = logging.getLogger(__name__)


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    gpu_memory = [int(x) for x in result.split()]
    return gpu_memory


def get_the_device_with_most_available_memory():#use_cuda_visible_devices=False):
    import torch
    if str(torch.__version__) == "0.4.1.":
        logger.warning("0.4.1. is bugged regarding mse loss")
    logger.info("Pytorch version : {}".format(torch.__version__))

    if not torch.cuda.is_available():
        device_str = "cpu"
    else:
        memory_map = get_gpu_memory_map()
        device_id = 0
        min = np.inf
        for k, v in enumerate(memory_map):
            logger.info("device={} memory used={}".format(k, v))
            # print type(v)
            if v < min:
                device_id = k
                min = v
        torch.cuda.set_device(device_id)
        device_str = "cuda:{}".format(device_id)
        # import torch
        # if use_cuda_visible_devices :
        #     # this seems to be the correct way to do it
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        #     device = "cuda"
        # else:
        #     # but this one is 2x faster when calling module.to(device)
        #     device = "cuda:{}".format(device)

    # print("importing torch ...")
    # import torch
    # print("done ...")
    # exit()

    device = torch.device(device_str)
    logger.info("device with most available memory: {}".format(device))
    return device
