from ncarrara.continuous_dqn.dqn.utils_dqn import run_dqn_with_transfer, run_dqn_without_transfer
from ncarrara.utils_rl.environments.envs_factory import generate_envs
from ncarrara.continuous_dqn.tools import utils
from ncarrara.continuous_dqn.tools.configuration import C

import numpy as np
import logging

from ncarrara.continuous_dqn.tools.features import build_feature_autoencoder, build_feature_dqn
from ncarrara.utils.math_utils import epsilon_decay

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import json


def main():
    epsilon_decay(C["transfer_dqn"]["start_decay"], C["transfer_dqn"]["decay"], C["transfer_dqn"]["N"], show=True)
    envs, tests_params = generate_envs(**C["target_envs"])
    feature_autoencoder = build_feature_autoencoder(C["feature_autoencoder_info"])
    feature_dqn = build_feature_dqn(C["feature_dqn_info"])
    source_params = C.load_sources_params()
    autoencoders = utils.load_autoencoders(C.path_models)
    ers = utils.load_memories(C.path_samples, C["create_data"]["as_json"])

    for er in ers:
        er.apply_feature_to_states(feature_dqn)
        er.to_tensors(C.device)

    N = C["transfer_dqn"]['N']

    results_w_t = np.zeros((len(envs), N))
    results_wo_t = np.zeros((len(envs), N))
    results_w_t_greedy = np.zeros((len(envs), N))
    results_wo_t_greedy = np.zeros((len(envs), N))
    print(results_w_t.shape)
    for i_env in range(len(envs)):
        test_params = tests_params[i_env]
        env = envs[i_env]
        logger.info("============================================================")
        logger.info("======================= ENV TARGET {} ======================".format(i_env))
        logger.info("============================================================")
        logger.info(test_params)

        logger.info("======== WITH TRANSFER ==========")

        r_w_t, r_w_t_greedy = run_dqn_with_transfer(env, seed=C.seed,
                                                    device=C.device,
                                                    autoencoders=autoencoders,
                                                    ers=ers,
                                                    sources_params=source_params,
                                                    test_params=test_params,
                                                    feature_autoencoder=feature_autoencoder,
                                                    feature_dqn=feature_dqn,
                                                    **C["transfer_dqn"])

        logger.info("======== WITHOUT TRANSFER ==========")
        r_wo_t, r_wo_t_greedy = run_dqn_without_transfer(env, seed=C.seed,
                                                         device=C.device,
                                                         sources_params=source_params,
                                                         feature_autoencoder=feature_autoencoder,
                                                         feature_dqn=feature_dqn,
                                                         test_params=test_params, **C["transfer_dqn"])

        results_w_t[i_env] = r_w_t
        results_wo_t[i_env] = r_wo_t
        results_w_t_greedy[i_env] = r_w_t_greedy
        results_wo_t_greedy[i_env] = r_wo_t_greedy
        show(tests_params,
             np.asarray(r_w_t),
             np.asarray(r_wo_t),
             np.asarray(r_w_t_greedy),
             np.asarray(r_wo_t_greedy),
             show_all=False)
        # exit()
    np.savetxt(C.path_results_w_t, results_w_t)
    np.savetxt(C.path_results_wo_t, results_wo_t)
    np.savetxt(C.path_results_w_t_greedy, results_w_t_greedy)
    np.savetxt(C.path_results_wo_t_greedy, results_wo_t_greedy)
    with open(C.path_targets_params, 'w') as file:
        dump = json.dumps(tests_params, indent=4)
        print(dump)
        file.write(dump)


def show(test_params=None, results_w_t=None, results_wo_t=None, results_w_t_greedy=None, results_wo_t_greedy=None,
         show_all=True):
    if test_params is None:
        test_params = C.load_targets_params()
    if results_w_t is None:
        results_w_t = np.loadtxt(C.path_results_w_t)
    if results_wo_t is None:
        results_wo_t = np.loadtxt(C.path_results_wo_t)
    if results_w_t_greedy is None:
        results_w_t_greedy = np.loadtxt(C.path_results_w_t_greedy)
    if results_wo_t_greedy is None:
        results_wo_t_greedy = np.loadtxt(C.path_results_wo_t_greedy)

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
        plt.title(C.id + " ienv={}".format(ienv))  # + "{:.2f}".format(test_params[ienv]['length']))
        plt.show()

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


if __name__ == "__main__":
    C.load("config/0_pydial.json").load_pytorch()
    main()
