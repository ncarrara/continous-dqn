from continuous_dqn.tools.configuration import C
import numpy as np
import logging
from continuous_dqn import dqn as utils_dqn

logger = logging.getLogger(__name__)
import gym

import matplotlib.pyplot as plt

def main():
    env = gym.make(C["source_envs"]['env_str'])
    N = C["no_transfer_dqn"]['N']
    rewards = utils_dqn.run_dqn_without_transfer(i_env=0,
                                                 env=env,
                                                 seed=C.seed,
                                                 env_params=C["source_envs"]['env_str'],
                                                 **C["no_transfer_dqn"])
    n_dots = 10
    xaxax =int(len(rewards) / int(N / n_dots))
    xxx = np.mean(np.reshape(np.array(rewards), (10, -1)), axis=1)
    print(xxx)
    plt.plot(range(len(xxx)), xxx)
    plt.title("DQN, no transfer")
    plt.show()
    plt.close()


if __name__ == "__main__":
    C.load("config/mountain_car_0.json")
    main()
