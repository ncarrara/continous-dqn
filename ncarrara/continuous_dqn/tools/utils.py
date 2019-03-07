import random
import numpy as np
import os
import torch
import logging

from ncarrara.continuous_dqn.tools.features import build_feature_autoencoder
from ncarrara.utils.color import Color
from ncarrara.utils_rl.transition.replay_memory import Memory

logger = logging.getLogger(__name__)


def load_models(path_models, device):
    files_models = os.listdir(path_models)
    autoencoders = [None] * len(files_models)
    for file in files_models:
        i_autoencoder = int(file.split(".")[0])
        path_model = path_models / file
        autoencoders[i_autoencoder] = torch.load(path_model, map_location=device)
    return autoencoders

def load_autoencoders(path_autoencoders,device):
    logger.info("reading autoencoders at {}".format(path_autoencoders))
    return load_models(path_autoencoders,device)

def load_q_sources(path_q_sources, device):
    logger.info("reading q sources at {}".format(path_q_sources))
    return load_models(path_q_sources, device)

def load_memories(path_data, as_json=True):
    logger.info("reading samples ...")
    files = os.listdir(path_data)
    logger.info("reading : {}".format(files))
    memories = [None] * len(files)
    if len(files) == 0:
        raise Exception("No data files in folder {}".format(path_data))
    for file in files:
        id_env = int(file.split(".")[0])
        path_file = path_data / file
        logger.info("reading {}".format(path_file))
        m = Memory()
        m.load_memory(path_file, as_json=as_json)
        memories[id_env] = m
    return memories


def read_samples_for_autoencoders(path_data, feature, device,as_json=True):
    from ncarrara.continuous_dqn.tools.configuration import C
    memories = load_memories(path_data, as_json=as_json)
    all_transitions = [None] * len(memories)
    for id_env, rm in enumerate(memories):
        data = np.array([feature(transition,device) for transition in rm.memory])
        all_transitions[id_env] = torch.from_numpy(data).float().to(C.device)
    return all_transitions


def array_to_cross_comparaison(tab, params_source, params_test):
    keys = params_source[0].keys()

    toprint = ""
    for ienv in range(len(tab)):
        formaterrors = format_errors(tab[ienv], params_source, params_test[ienv], show_params=True) + "\n"
        toprint += formaterrors

    len_params = len("".join([v+" " if type(v) == str else "{:.2f} ".format(v) for v in params_test[0].values()])) + 2

    head = ""  # ""-" * (6+len_params) * len(params_source) + "\n"
    for key in keys:
        xx = " " * len_params
        for param in params_source:
            xx += param[key]+" " if type(param[key]) == str else "{:5.2f} ".format(param[key])
        head += "{} <- {}\n".format(xx, key)
    head = head + " " * len_params + "-" * 6 * len(params_source) + "\n"

    return head + toprint


def format_errors(errors, params_source, param_test, show_params=False):
    toprint = "" if not show_params else "".join([v+" " if type(v) == str else "{:.2f} ".format(v) for v in param_test.values()]) + "| "
    min_idx = np.argmin(errors)
    print(errors)
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
