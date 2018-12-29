import numpy as np
import logging
import random
import torch
logger = logging.getLogger(__name__)


def set_seed(seed):
    if seed is not None:
        logger.info("Setting seed = {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def epsilon_decay(start=1.0, decay=0.01, N=100):
    return [np.exp(-n / (1. / decay)) * start for n in range(N)]

def normalized(a):
    sum = 0.
    for v in a:
        sum += v
    rez = np.array([v / sum for v in a])
    return rez

# TODO check if ok
def create_arrangements(nb_elements, size_arr, current_size_arr=0, arrs=None):
    new_arrs = []
    if not arrs:
        arrs = [[]]
    for arr in arrs:
        for i in range(0, nb_elements):
            new_arr = list(arr)
            new_arr.append(i)
            new_arrs.append(new_arr)
    if current_size_arr >= size_arr - 1:
        return new_arrs
    else:
        return create_arrangements(nb_elements=nb_elements,
                                   size_arr=size_arr,
                                   current_size_arr=current_size_arr + 1,
                                   arrs=new_arrs)