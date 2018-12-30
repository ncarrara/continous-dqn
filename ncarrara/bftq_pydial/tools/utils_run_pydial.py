import numpy as np

import ncarrara.bftq_pydial.bftq.pytorch_budgeted_fittedq as pbf
from ncarrara.utils.datastructure import merge_two_dicts
from ncarrara.utils_rl.transition.transition import Transition


def datas_to_transitions(datas, env, feature, lambda_, normalize_reward):
    print("data to transition ... ")
    max_r_ftq = 0
    max_r_bftq = 0
    e = env
    nbNone = 0
    for data in datas:
        # if not data.a in 'hello()':
        reward_ftq = data.r_ - (lambda_ * data.info["c_"])
        max_r_ftq = np.abs(reward_ftq) if np.abs(reward_ftq) > max_r_ftq else max_r_ftq
        max_r_bftq = np.abs(data.r_) if np.abs(data.r_) > max_r_bftq else max_r_bftq
        if data.s_ is None: nbNone += 1
        if data.s_ is not None and (data.r_ == 20. or data.r_ == -20): raise Exception("not cool bro")

    transitions_ftq = []
    transitions_bftq = []
    for data in datas:
        # if not data.a in 'hello()':
        r_ = data.r_
        s = feature(data.s, e)
        s_ = feature(data.s_, e)
        a = data.a #e.action_space().index(data.a)
        c_ = data.info["c_"]
        reward_ftq = r_ - (lambda_ * c_)
        reward_bftq = r_
        if normalize_reward:
            reward_ftq /= max_r_ftq
            reward_bftq /= max_r_bftq
        t_ftq = Transition(s, a, reward_ftq, s_)
        t_bftq = pbf.TransitionBFTQ(s, a, reward_bftq, s_, c_, None, None)
        transitions_ftq.append(t_ftq)
        transitions_bftq.append(t_bftq)
    print("nbdialogues : ", nbNone)
    print("data to transition ... done")
    return transitions_ftq, transitions_bftq


def print_results(results):
    N = len(results)
    rew_r, rew_c, ret_r, ret_c = np.mean(results, axis=0)
    std_rew_r, std_rew_c, std_ret_r, std_ret_c = np.std(results, axis=0)
    p = "R={:.2f}+/-{:.2f} C={:.2f}+/-{:.2f} , return : R={:.2f}+/-{:.2f} C={:.2f}+/-{:.2f}".format(
        rew_r, std_rew_r, rew_c, std_rew_c, ret_r, std_ret_r, ret_c, std_ret_c)
    confidence_r = 2 * (std_rew_r / np.sqrt(N))
    confidence_r_str = "[{:.2f};{:.2f}]".format(rew_r - confidence_r, rew_r + confidence_r)
    confidence_c = 2 * (std_rew_c / np.sqrt(N))
    confidence_c_str = "[{:.2f};{:.2f}]".format(rew_c - confidence_c, rew_c + confidence_c)
    pp = "R=" + confidence_r_str + " C=" + confidence_c_str
    print(pp + " " + p)



def execute_policy_one_dialogue(env, pi, gamma_r=1.0, gamma_c=1.0, beta=1.0, print_dial=False):
    if print_dial:
        print('---------------------------------')
    dialogue = []
    pi.reset()
    s = env.reset()
    a= env.action_space_str.index('hello')
    rew_r, rew_c, ret_r, ret_c = 0., 0., 0., 0.
    i = 0
    s_, r_, end, info_env = env.step(a)
    if print_dial: print(a)
    turn = (s, a, r_, s_, end, info_env)
    dialogue.append(turn)
    info_env = {}
    info_pi = {"beta": beta}
    i += 1

    while not end:
        s = s_
        actions = env.action_space_executable()
        action_mask=np.zeros(env.action_space.n)
        for action in actions:
            action_mask[action]=1
        info_pi = merge_two_dicts(info_pi, info_env)
        a, is_master_action, info_pi = pi.execute(s, action_mask, info_pi)
        s_, r_, end, info_env = env.step(a, is_master_act=is_master_action)
        if print_dial:
            print("i={} ||| sys={} ||| usr={} ||| patience={}".format(i, a,
                                                                      info_env["last user act"],
                                                                      info_env["patience"]))
        c_ = info_env["c_"]  # 1. / env.maxTurns  # info_env["c_"]
        # print s_
        turn = (s, int(a), r_, s_, end, info_env)
        rew_r += r_
        rew_c += c_
        ret_r += r_ * (gamma_r ** i)
        ret_c += c_ * (gamma_c ** i)
        dialogue.append(turn)
        last_a = a
        i += 1

    return dialogue, rew_r, rew_c, ret_r, ret_c


def execute_policy(env, pi, gamma_r=1.0, gamma_c=1.0, N_dialogues=10, beta=1., print_dial=False):
    dialogues = []
    result = np.zeros((N_dialogues, 4))
    turn = 0
    for d in range(N_dialogues):
        dialogue, rew_r, rew_c, ret_r, ret_c = execute_policy_one_dialogue(env, pi, gamma_r, gamma_c, beta,
                                                                           print_dial=print_dial)
        dialogues.append(dialogue)
        result[d] = np.array([rew_r, rew_c, ret_r, ret_c])
        turn += len(dialogue)
    print("mean turn : ", turn / float(N_dialogues))
    return dialogues, result


#
# def change_trajectories(M, trajectories):
#     res = []
#     for trajectory in trajectories:
#         t = []
#         for sample in trajectory:
#             s, a, rp, sp, cp = sample
#             if cp > 0.:
#                 rp = -M
#             t.append((s, a, rp, sp, cp))
#         res.append(t)
#     return res
#
#
# def change_samples(M, samples):
#     res = []
#     for sample in samples:
#         s, a, rp, sp, cp = sample
#         if cp > 0.:
#             rp = -M
#         res.append((s, a, rp, sp, cp))
#     return res

