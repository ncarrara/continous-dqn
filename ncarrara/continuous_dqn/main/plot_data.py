from ncarrara.continuous_dqn.dqn.utils_dqn import run_dqn
from ncarrara.utils.os import makedirs
from ncarrara.utils_rl.environments.envs_factory import generate_envs
from ncarrara.continuous_dqn.tools import utils

import numpy as np
import logging

from ncarrara.continuous_dqn.tools.features import build_feature_autoencoder, build_feature_dqn
from ncarrara.utils.math_utils import epsilon_decay

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import json


def main(results_w_t=None, results_wo_t=None, results_w_t_greedy=None, results_wo_t_greedy=None, workspace=None,
         show_all=True):
    makedirs(workspace / "plots")
    if results_w_t is None:
        results_w_t = np.loadtxt(workspace / "target" / "w_transfer" / "results")
    if results_wo_t is None:
        results_wo_t = np.loadtxt(workspace / "target" / "wo_transfer" / "results")
    if results_w_t_greedy is None:
        results_w_t_greedy = np.loadtxt(workspace / "target" / "w_transfer" / "results_greedy")
    if results_wo_t_greedy is None:
        results_wo_t_greedy = np.loadtxt(workspace / "target" / "wo_transfer" / "results_greedy")

    if results_w_t.ndim > 1:
        N_trajs = results_w_t.shape[1]
        N_envs = results_w_t.shape[0]
    else:
        N_trajs = results_w_t.shape[0]
        N_envs = 1
    # print(results_w_t.shape)
    # print(N_trajs)
    # exit()

    n_dots = min(5, N_trajs)
    traj_by_dot = int(N_trajs / n_dots)

    x = np.linspace(1, n_dots, n_dots) * traj_by_dot

    for ienv in range(N_envs):
        if results_w_t.ndim > 1:
            r_w_t = results_w_t[ienv]
            r_wo_t = results_wo_t[ienv]
            r_wo_t_greedy = results_wo_t_greedy[ienv]
            r_w_t_greedy = results_w_t_greedy[ienv]
        else:
            r_w_t = results_w_t
            r_wo_t = results_wo_t
            r_wo_t_greedy = results_wo_t_greedy
            r_w_t_greedy = results_w_t_greedy

        r_wo_t = np.mean(np.reshape(np.array(r_wo_t), (int(len(r_wo_t) / traj_by_dot), -1)), axis=1)
        r_wo_t_greedy = np.mean(
            np.reshape(np.array(r_wo_t_greedy), (int(len(r_wo_t_greedy) / traj_by_dot), -1)),
            axis=1)
        r_w_t = np.mean(np.reshape(np.array(r_w_t), (int(len(r_w_t) / traj_by_dot), -1)), axis=1)
        r_w_t_greedy = np.mean(np.reshape(np.array(r_w_t_greedy), (int(len(r_w_t_greedy) / traj_by_dot), -1)),
                               axis=1)

        wo_transfer, = plt.plot(x, r_wo_t, label="w/o transfert", color='blue')
        wo_transfer_greedy, = plt.plot(x, r_wo_t_greedy, label="w/o transfert (greedy)", color='blue', marker='*',
                                       markersize=15)
        w_transfer, = plt.plot(x, r_w_t, label="with transfert", color='orange')
        w_transfer_greedy, = plt.plot(x, r_w_t_greedy, label="with transfert (greedy)",
                                      color='orange', marker='*', markersize=15)
        plt.legend(handles=[wo_transfer, wo_transfer_greedy, w_transfer, w_transfer_greedy])
        plt.title("ienv={}".format(ienv))  # + "{:.2f}".format(test_params[ienv]['length']))
        plt.show()
        plt.savefig(workspace / "plots" / "ienv={}".format(ienv))

        plt.close()

    def smooth(data):
        data = np.mean(data, axis=0)
        data = np.mean(np.reshape(np.array(data), (int(len(data) / traj_by_dot), -1)), axis=1)
        return data

    if N_envs > 1 and show_all:
        # x = np.array(range(n_dots)) * (int(N_trajs / n_dots))
        data = smooth(results_wo_t)
        wo_transfer, = plt.plot(x, data, label="wo transfert", color='blue')

        data = smooth(results_wo_t_greedy)
        wo_transfer_greedy, = plt.plot(x, data, label="wo transfert (greedy)", color='blue', marker='*',
                                       markersize=15)

        data = smooth(results_w_t)
        w_transfer, = plt.plot(x, data, label="with transfert", color='orange')

        data = smooth(results_w_t_greedy)
        w_transfer_greedy, = plt.plot(x, data, label="with transfert (greedy)", color='orange',
                                      marker='*',
                                      markersize=15)

        plt.legend(handles=[wo_transfer, wo_transfer_greedy, w_transfer, w_transfer_greedy])
        plt.title("all")
        plt.show()
