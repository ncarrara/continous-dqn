from configuration import C
import utils
import random
import numpy as np
from dqn.transition import Transition
from envs.envs_factory import generate_envs
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)
import dqn.utils_dqn as utils_dqn
import matplotlib.pyplot as plt
import gym


def main():
    envs, params = generate_envs(**C["generate_samples"])
    autoencoders = utils.load_autoencoders(C.workspace + "/" + C.path_models)
    ers = utils.load_experience_replays(C.workspace + "/" + C.path_samples)
    for er in ers:
        er.to_tensors()

    N = C["transfer_dqn"]['N']

    for i_env in range(len(envs)):
        param = params[i_env]
        env = envs[i_env]
        print(param)

        r_w_t, r_w_t_greedy = utils_dqn.run_dqn_with_transfer(i_env, env, seed=C.seed,
                                                              autoencoders=autoencoders,
                                                              ers=ers,
                                                              env_params=params,
                                                              **C["transfer_dqn"])

        r_wo_t, r_wo_t_greedy = utils_dqn.run_dqn_without_transfer(i_env, env, seed=C.seed,
                                                                   env_params=params, **C["transfer_dqn"])

        n_dots = 10

        r_wo_t = np.mean(np.reshape(np.array(r_wo_t), (int(len(r_wo_t) / int(N / n_dots)), -1)), axis=1)
        r_wo_t_greedy = np.mean(np.reshape(np.array(r_wo_t_greedy), (int(len(r_wo_t_greedy) / int(N / n_dots)), -1)),
                                axis=1)
        r_w_t = np.mean(np.reshape(np.array(r_w_t), (int(len(r_w_t) / int(N / n_dots)), -1)), axis=1)
        r_w_t_greedy = np.mean(np.reshape(np.array(r_w_t_greedy), (int(len(r_w_t_greedy) / int(N / n_dots)), -1)),
                               axis=1)

        wo_transfer, = plt.plot(range(len(r_wo_t)), r_wo_t, label="w/o transfert", color='blue')
        wo_transfer_greedy, = plt.plot(range(len(r_wo_t_greedy)), r_wo_t_greedy, label="w/o transfert (greedy)",
                                       color='blue', marker='*', markersize=15)
        w_transfer, = plt.plot(range(len(r_w_t)), r_w_t, label="with transfert", color='orange')
        w_transfer_greedy, = plt.plot(range(len(r_w_t_greedy)), r_w_t_greedy, label="with transfert (greedy)",
                                      color='orange', marker='*', markersize=15)
        plt.legend(handles=[wo_transfer, wo_transfer_greedy, w_transfer, w_transfer_greedy])
        plt.title(C.id + " " + str(param['length']))
        plt.show()

        plt.close()


if __name__ == "__main__":
    C.load("config/0.json")
    main()
