from dqn.transfer_module import TransferModule
from configuration import C
import utils
import random
import numpy as np
from dqn.transition import Transition
from envs.envs_factory import generate_envs
import torch.nn.functional as F
import logging
from dqn.dqn import DQN
from dqn.net import Net
import matplotlib.pyplot as plt
import dqn.utils_dqn as utils_dqn

logger = logging.getLogger(__name__)
import gym


def main():
    env = gym.make("CartPole-v0")
    N=C["transfer_dqn"]['N']

    # net_params = C["no_transfer_dqn"]["net"]
    # dqn_params = C["no_transfer_dqn"]["dqn"]
    # decay = C["no_transfer_dqn"]["decay"]
    # N = C["no_transfer_dqn"]["N"]

    # rewards = utils_dqn.run_dqn_without_transfer(env,decay=decay,dqn_params=dqn_params,net_params=net_params,N=N)
    rewards = utils_dqn.run_dqn_without_transfer(env, seed=C.seed, **C["no_transfer_dqn"])
    n_dots = 10
    xxx = np.mean(np.reshape(np.array(rewards), (int(len(rewards) / int(N / n_dots)), -1)), axis=1)
    plt.plot(range(len(xxx)), xxx)


if __name__ == "__main__":
    C.load("config/0.json")
    main()
