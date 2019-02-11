import numpy as np
import logging
import random
import matplotlib.pyplot as plt

from ncarrara.utils.os import makedirs

logger = logging.getLogger(__name__)

def generate_random_point_on_simplex_not_uniform(coeff, bias, min_x, max_x):
    """
    Warning, this is not uniform sampling. Need to found a way to not favorise the vertex (maybe look at dirichlet distribution)
    :param coeff:
    :param bias:
    :param min_x:
    :param max_x:
    :return:
    """
    x = np.zeros(len(coeff))
    indexes = np.asarray(range(0, len(coeff)))
    np.random.shuffle(indexes)  # need shuffle to not favorise a dimension
    remain_indexes = np.copy(indexes)
    for i_index, index in enumerate(indexes):
        remain_indexes = remain_indexes[1:]
        current_coeff = np.take(coeff, remain_indexes)
        fullmin = np.full(len(remain_indexes), min_x)
        fullmax = np.full(len(remain_indexes), max_x)
        dotmax = np.dot(current_coeff, fullmax)
        dotmin = np.dot(current_coeff, fullmin)
        min_xi = (bias - dotmax) / coeff[index]
        max_xi = (bias - dotmin) / coeff[index]
        min_xi = np.max([min_xi, min_x])
        max_xi = np.min([max_xi, max_x])
        xi = min_xi + np.random.random_sample() * (max_xi - min_xi)
        bias = bias - xi * coeff[index]
        x[index] = xi
        if len(remain_indexes) == 1:
            break
    last_index = remain_indexes[0]
    x[last_index] = bias / coeff[last_index]
    return x

def to_onehot(vector, max_value):
    rez = [0] * (max_value + 1)
    for i in range(len(vector)):
        index = vector[i] if vector[i] < max_value else max_value
        index = i * (max_value + 1) + index
        rez[int(index)] = 1.
    return rez

def set_seed(seed, env=None):
    if seed is not None:
        logger.info("Setting seed = {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)

        if env is not None:
            env.seed(seed)
            env.reset()

def epsilon_decay(start=1.0, decay=0.01, N=100,savepath=None):
    makedirs(savepath)
    decays = [np.exp(-n / (1. / decay)) * start for n in range(N)]
    logger.info("Epsilons (decayed) : [{}]".format(''.join(["{:.2f} ".format(eps) for eps in decays])))
    if logger.getEffectiveLevel() is logging.INFO:
        plt.plot(range(len(decays)),decays)
        plt.title("epsilon decays")
        plt.show()
        if savepath is not None:
            plt.savefig(savepath+"/epsilon_decay")
        plt.close()
    return decays

def normalized(a):
    sum = 0.
    for v in a:
        sum += v
    rez = np.array([v / sum for v in a])
    return rez

def update_lims(lims, values):
    return min(lims[0], np.amin(values)), max(lims[1], np.amax(values))

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

if __name__=="__main__":
    epsilon_decay(1.0,0.0005,10000,show=True)