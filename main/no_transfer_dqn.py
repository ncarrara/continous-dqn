from configuration import C
import numpy as np
import logging
import matplotlib.pyplot as plt
import dqn.utils_dqn as utils_dqn

logger = logging.getLogger(__name__)
import gym


def main():
    env = gym.make(C["no_transfer_dqn"]['env_str'])
    N = C["no_transfer_dqn"]['N']
    rewards = utils_dqn.run_dqn_without_transfer(env=env,
                                                 seed=C.seed,
                                                 env_params=C["no_transfer_dqn"]['env_str'],
                                                 **C["no_transfer_dqn"])
    n_dots = 10
    xxx = np.mean(np.reshape(np.array(rewards), (int(len(rewards) / int(N / n_dots)), -1)), axis=1)
    plt.plot(range(len(xxx)), xxx)


if __name__ == "__main__":
    C.load("config/mountain_car_0.json")
    main()
