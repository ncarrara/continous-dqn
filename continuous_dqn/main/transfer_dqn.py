from continuous_dqn.dqn.utils_dqn import run_dqn_with_transfer, run_dqn_without_transfer
from continuous_dqn.envs.envs_factory import generate_envs
from continuous_dqn.tools.configuration import C
from continuous_dqn.tools import utils
import numpy as np
import logging

logger = logging.getLogger(__name__)
from continuous_dqn import dqn as utils_dqn
import matplotlib.pyplot as plt
import json


def main():
    envs, tests_params = generate_envs(**C["target_envs"])
    source_params = C.load_sources_params()
    autoencoders = utils.load_autoencoders(C.path_models)
    ers = utils.load_experience_replays(C.path_samples)
    for er in ers:
        er.to_tensors(C.device)

    N = C["transfer_dqn"]['N']

    results_w_t = np.zeros((len(envs), N))
    results_wo_t = np.zeros((len(envs), N))
    results_w_t_greedy = np.zeros((len(envs), N))
    results_wo_t_greedy = np.zeros((len(envs), N))

    for i_env in range(len(envs)):
        test_params = tests_params[i_env]
        env = envs[i_env]
        logger.info("============================================================")
        logger.info("======================= ENV TARGET {} ======================".format(i_env))
        logger.info("============================================================")
        logger.info(test_params)

        logger.info("======== WITH TRANSFER ==========")

        r_w_t, r_w_t_greedy = run_dqn_with_transfer(env, seed=C.seed,
                                                              autoencoders=autoencoders,
                                                              ers=ers,
                                                              sources_params=source_params,
                                                              test_params=test_params,
                                                              **C["transfer_dqn"])

        logger.info("======== WITHOUT TRANSFER ==========")
        r_wo_t, r_wo_t_greedy = run_dqn_without_transfer(env, seed=C.seed,
                                                                   sources_params=source_params,
                                                                   test_params=test_params, **C["transfer_dqn"])

        results_w_t[i_env] = r_w_t
        results_wo_t[i_env] = r_wo_t
        results_w_t_greedy[i_env] = r_w_t_greedy
        results_wo_t_greedy[i_env] = r_wo_t_greedy
    np.savetxt(C.path_results_w_t, results_w_t)
    np.savetxt(C.path_results_wo_t, results_wo_t)
    np.savetxt(C.path_results_w_t_greedy, results_w_t_greedy)
    np.savetxt(C.path_results_wo_t_greedy, results_wo_t_greedy)
    with open(C.path_targets_params, 'w') as file:
        dump = json.dumps(tests_params, indent=4)
        print(dump)
        file.write(dump)


def show():
    test_params = C.load_targets_params()
    results_w_t = np.loadtxt(C.path_results_w_t)
    results_wo_t = np.loadtxt(C.path_results_wo_t)
    results_w_t_greedy = np.loadtxt(C.path_results_w_t_greedy)
    results_wo_t_greedy = np.loadtxt(C.path_results_wo_t_greedy)

    n_dots = 10

    N_trajs = results_w_t.shape[1]
    N_envs = results_w_t.shape[0]

    for ienv in range(N_envs):
        r_w_t = results_w_t[ienv]
        r_wo_t = results_wo_t[ienv]
        r_wo_t_greedy = results_wo_t_greedy[ienv]
        r_w_t_greedy = results_w_t_greedy[ienv]

        r_wo_t = np.mean(np.reshape(np.array(r_wo_t), (int(len(r_wo_t) / int(N_trajs / n_dots)), -1)), axis=1)
        r_wo_t_greedy = np.mean(
            np.reshape(np.array(r_wo_t_greedy), (int(len(r_wo_t_greedy) / int(N_trajs / n_dots)), -1)),
            axis=1)
        r_w_t = np.mean(np.reshape(np.array(r_w_t), (int(len(r_w_t) / int(N_trajs / n_dots)), -1)), axis=1)
        r_w_t_greedy = np.mean(np.reshape(np.array(r_w_t_greedy), (int(len(r_w_t_greedy) / int(N_trajs / n_dots)), -1)),
                               axis=1)

        wo_transfer, = plt.plot(range(len(r_wo_t)), r_wo_t, label="w/o transfert", color='blue')
        wo_transfer_greedy, = plt.plot(range(len(r_wo_t_greedy)), r_wo_t_greedy, label="w/o transfert (greedy)",
                                       color='blue', marker='*', markersize=15)
        w_transfer, = plt.plot(range(len(r_w_t)), r_w_t, label="with transfert", color='orange')
        w_transfer_greedy, = plt.plot(range(len(r_w_t_greedy)), r_w_t_greedy, label="with transfert (greedy)",
                                      color='orange', marker='*', markersize=15)
        plt.legend(handles=[wo_transfer, wo_transfer_greedy, w_transfer, w_transfer_greedy])
        plt.title(C.id + "{:.2f}".format(test_params[ienv]['length']))
        plt.show()

        plt.close()

    def smooth(data):
        data = np.mean(data, axis=0)
        data = np.mean(np.reshape(np.array(data), (int(len(data) / int(N_trajs / n_dots)), -1)), axis=1)
        return data

    data = smooth(results_wo_t)
    wo_transfer, = plt.plot(range(len(data)), data, label="wo transfert", color='blue')

    data = smooth(results_wo_t_greedy)
    wo_transfer_greedy, = plt.plot(range(len(data)), data, label="wo transfert (greedy)", color='blue', marker='*',
                                   markersize=15)

    data = smooth(results_w_t)
    w_transfer, = plt.plot(range(len(data)), data, label="with transfert", color='orange')

    data = smooth(results_w_t_greedy)
    w_transfer_greedy, = plt.plot(range(len(data)), data, label="with transfert (greedy)", color='orange', marker='*',
                                  markersize=15)

    plt.legend(handles=[wo_transfer, wo_transfer_greedy, w_transfer, w_transfer_greedy])
    plt.title("all")
    plt.show()


if __name__ == "__main__":
    C.load("config/0_random.json")
    main()
