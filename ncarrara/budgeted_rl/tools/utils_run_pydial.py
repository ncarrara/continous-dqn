import numpy as np

import ncarrara.budgeted_rl.bftq.pytorch_budgeted_fittedq as pbf
from ncarrara.utils.datastructure import merge_two_dicts
from ncarrara.utils.os import makedirs
from ncarrara.utils_rl.transition.transition import Transition

import logging

logger = logging.getLogger(__name__)


def datas_to_transitions(datas, env, feature, lambda_, normalize_reward):
    logger.info("data to transition ... ")
    max_r_ftq = 0
    max_r_bftq = 0
    e = env
    nbNone = 0
    for data in datas:
        reward_ftq = data.r_ - (lambda_ * data.info["c_"])
        max_r_ftq = np.abs(reward_ftq) if np.abs(reward_ftq) > max_r_ftq else max_r_ftq
        max_r_bftq = np.abs(data.r_) if np.abs(data.r_) > max_r_bftq else max_r_bftq
        if data.s_ is None: nbNone += 1
    logger.info("max_r_ftq : {:.2f} max_r_bftq : {:.2f}".format(max_r_ftq, max_r_bftq))
    transitions_ftq = []
    transitions_bftq = []
    for data in datas:
        # if not data.a in 'hello()':
        r_ = data.r_
        beta = data.info["beta"]
        s = feature(data.s, e)
        if data.done:
            s_ = None
        else:
            s_ = feature(data.s_, e)
        a = data.a  # e.action_space().index(data.a)
        c_ = data.info["c_"]
        reward_ftq = r_ - (lambda_ * c_)
        reward_bftq = r_
        if normalize_reward:
            reward_ftq /= (1. if max_r_ftq == 0. else max_r_ftq)
            reward_bftq /= (1. if max_r_bftq == 0. else max_r_bftq)
        t_ftq = Transition(s, a, reward_ftq, s_)
        t_bftq = pbf.TransitionBFTQ(s, a, reward_bftq, s_, c_, beta, None)
        transitions_ftq.append(t_ftq)
        transitions_bftq.append(t_bftq)
    logger.info("nbdialogues : {}".format(nbNone))
    logger.info("data to transition ... done")
    return transitions_ftq, transitions_bftq


def format_results(results):
    N = len(results)
    rew_r, rew_c, ret_r, ret_c = np.mean(results, axis=0)
    std_rew_r, std_rew_c, std_ret_r, std_ret_c = np.std(results, axis=0)
    p = "R={:.2f}+/-{:.2f} C={:.2f}+/-{:.2f} , return : R={:.2f}+/-{:.2f} C={:.2f}+/-{:.2f}".format(
        rew_r, std_rew_r, rew_c, std_rew_c, ret_r, std_ret_r, ret_c, std_ret_c)
    confidence_r = 1.96 * (std_rew_r / np.sqrt(N))
    confidence_r_str = "[{:.2f};{:.2f}]".format(rew_r - confidence_r, rew_r + confidence_r)
    confidence_c = 1.96 * (std_rew_c / np.sqrt(N))
    confidence_c_str = "[{:.2f};{:.2f}]".format(rew_c - confidence_c, rew_c + confidence_c)
    pp = "R=" + confidence_r_str + " C=" + confidence_c_str
    return (pp + " " + p)


def execute_policy_one_dialogue(env, pi, gamma_r=1.0, gamma_c=1.0, beta=None):
    dialogue = []
    pi.reset()

    if hasattr(env, "ID") and env.ID == "gym_pydial":
        s = env.reset()
        a = env.action_space_str.index('hello')
        rew_r, rew_c, ret_r, ret_c = 0., 0., 0., 0.
        i = 0
        s_, r_, end, info_env = env.step(a)
        turn = (s, a, r_, s_, end, info_env)
        dialogue.append(turn)
        info_env = {}
        info_pi = {"beta": beta}
        i += 1
    else:
        s_ = env.reset()
        rew_r, rew_c, ret_r, ret_c = 0., 0., 0., 0.
        i = 0
        info_env = {}
        info_pi = {"beta": beta}
        end = False

    while not end:
        s = s_
        action_mask = [0.] * env.action_space.n
        if hasattr(env, "action_space_executable"):
            # print("action_space_executable !!!")
            raise Exception("Remove this expection please")
            actions = env.action_space_executable()
            action_mask = np.zeros(env.action_space.n)
            for action in actions:
                action_mask[action] = 1

        beta = info_pi["beta"]

        info_pi = merge_two_dicts(info_pi, info_env)


        a, is_master_action, info_pi = pi.execute(s, action_mask, info_pi)
        if hasattr(env, "ID") and env.ID == "gym_pydial":
            s_, r_, end, info_env = env.step(a, is_master_act=is_master_action)
        else:
            s_, r_, end, info_env = env.step(a)
        c_ = info_env["c_"]

        info = {**info_env}
        info["beta"] = beta

        turn = (s, a if type(a) is str else int(a), r_, s_, end, info)
        rew_r += r_
        rew_c += c_
        ret_r += r_ * (gamma_r ** i)
        ret_c += c_ * (gamma_c ** i)
        dialogue.append(turn)
        i += 1

    return dialogue, rew_r, rew_c, ret_r, ret_c


def execute_policy(env, pi, gamma_r=1.0, gamma_c=1.0, N_dialogues=10, beta=1., save_path=None):
    dialogues = []
    result = np.zeros((N_dialogues, 4))
    turn = 0
    for d in range(N_dialogues):
        dialogue, rew_r, rew_c, ret_r, ret_c = execute_policy_one_dialogue(env, pi, gamma_r, gamma_c, beta)
        dialogues.append(dialogue)
        result[d] = np.array([rew_r, rew_c, ret_r, ret_c])
        turn += len(dialogue)
    logger.info("[execute_policy] mean turn : {}".format(turn / float(N_dialogues)))
    if save_path is not None:
        logger.info("[execute_policy] saving results at : {}".format(save_path))
        np.savetxt(save_path, result)
    return dialogues, result
