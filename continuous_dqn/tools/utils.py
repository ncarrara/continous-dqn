import random
import numpy as np
import os
import torch
import logging

from utils.color import Color
from utils_rl.transition.replay_memory import ReplayMemory
from utils_rl.transition.transition import TransitionGym

logger = logging.getLogger(__name__)


def load_autoencoders(path_autoencoders):
    from continuous_dqn.tools.configuration import C
    logger.info("reading autoencoders at {}".format(path_autoencoders))
    files_autoencoders = os.listdir(path_autoencoders)
    autoencoders = [None] * len(files_autoencoders)
    for file in files_autoencoders:
        i_autoencoder = int(file.split(".")[0])
        path_autoencoder = path_autoencoders + "/" + file
        autoencoders[i_autoencoder] = torch.load(path_autoencoder, map_location=C.device)
    return autoencoders


def load_experience_replays(path_data):
    logger.info("reading samples ...")
    files = os.listdir(path_data)
    logger.info("reading : {}".format(files))
    ers = [None] * len(files)
    if len(files)==0:
        raise Exception("No data files in folder {}".format(path_data))
    for file in files:
        id_env = int(file.split(".")[0])
        path_file = path_data + "/" + file
        logger.info("reading {}".format(path_file))
        rm = ReplayMemory(10000,TransitionGym)
        rm.load_memory(path_file)
        ers[id_env] = rm
    return ers


def read_samples(path_data):
    from continuous_dqn.tools.configuration import C
    ers = load_experience_replays(path_data)
    all_transitions = [None] * len(ers)
    for id_env, rm in enumerate(ers):
        data = rm.sample_to_numpy(len(rm))
        all_transitions[id_env] = torch.from_numpy(data).float().to(C.device)
    return all_transitions


def array_to_cross_comparaison(tab, params_source, params_test):
    keys = params_source[0].keys()

    toprint = ""
    for ienv in range(len(tab)):
        formaterrors = format_errors(tab[ienv], params_source, params_test[ienv],show_params=True) + "\n"
        toprint += formaterrors

    len_params = len("".join(["{:.2f} ".format(v) for v in params_test[0].values()]))+2

    head = "" #""-" * (6+len_params) * len(params_source) + "\n"
    for key in keys:
        xx = " " * len_params
        for param in params_source:
            xx += "{:5.2f} ".format(param[key])
        head += "{} <- {}\n".format(xx, key)
    head = head + " "*len_params+ "-" * 6 * len(params_source) + "\n"

    return head + toprint


def format_errors(errors, params_source, param_test,show_params=False):
    toprint = "" if not show_params else "".join(["{:.2f} ".format(v) for v in param_test.values()])+"| "
    min_idx = np.argmin(errors)

    for isource in range(len(errors)):
        same_env = params_source[isource] == param_test

        if isource == min_idx and same_env:
            toprint += Color.UNDERLINE + Color.BOLD + Color.PURPLE + "{:5.2f} ".format(
                errors[isource]) + Color.END + Color.END + Color.END
        elif isource == min_idx and not same_env:
            toprint += Color.UNDERLINE + Color.BOLD + "{:5.2f} ".format(errors[isource]) + Color.END + Color.END
        elif isource != min_idx and same_env:
            toprint += Color.PURPLE + "{:5.2f} ".format(errors[isource]) + Color.END
        else:
            toprint += "{:5.2f} ".format(errors[isource])

    diff = ""
    if param_test != params_source[min_idx]:
        for k in param_test.keys():
            va = param_test[k]
            vb = params_source[min_idx][k]
            if va != vb:
                value_env = Color.PURPLE + "{:.2f}".format(va) + Color.END
                value_better = Color.UNDERLINE + Color.BOLD + "{:.2f}".format(vb) + Color.END + Color.END
                diff += k + ":" + value_env + "/" + value_better + " "
    return toprint + "\t" + diff


def set_seed(seed, env=None):
    if seed is not None:
        logger.info("Setting seed = {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if env is not None:
            env.seed(seed)

