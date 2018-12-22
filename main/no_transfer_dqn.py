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
logger = logging.getLogger(__name__)
import gym


def main():
    N = 1000
    # envs, params = generate_envs(**C["generate_samples"])
    env = gym.make("CartPole-v0")
    net = Net(**C["net"])
    dqn = DQN(policy_network=net, **C["dqn"])
    dqn.reset()
    utils.set_seed(seed=C.seed, env=env)
    env.seed(C.seed)
    # render = lambda: plt.imshow(env.render(mode='rgb_array'))

    rrr=[]
    epsilons = utils.epsilon_decay(start=1.0,decay=0.01,N=N)
    plt.plot(range(N),epsilons)
    plt.show()
    for n in range(N):
        s = env.reset()
        done = False
        rr=0
        while not done:

            if random.random() < epsilons[n]:
                a = env.action_space.sample()
            else:
                a = dqn.pi(s, np.zeros(env.action_space.n))
            s_, r_, done, info = env.step(a)
            rr+=r_
            t = (s.tolist(), a, r_, s_.tolist())
            dqn.update(*t)
            s = s_
        rrr.append(rr)
        n_dots=10
        if len(rrr)%int(N/n_dots)==0 and len(rrr)>0:
            print("n",n)
            xxx = np.mean(np.reshape(np.array(rrr),(int(len(rrr)/int(N/n_dots)),-1)),axis=1)
            plt.plot(range(len(xxx)),xxx)
            plt.show()


if __name__ == "__main__":
    # execute only if run as a script
    main()
