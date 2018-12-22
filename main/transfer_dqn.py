from dqn.transfer_module import TransferModule
from configuration import C
import utils
import numpy as np
from dqn.transition import Transition
from envs.envs_factory import generate_envs
import torch.nn.functional as F
import logging
from dqn.dqn import DQN
from dqn.net import Net
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


def main():
    N = 100
    envs, params = generate_envs(**C["generate_samples"])
    env = envs[0]
    net = Net(**C["net"])
    dqn = DQN(policy_network=net, **C["dqn"])

    autoencoders = utils.load_autoencoders(C.workspace + "/" + C.path_models)
    # all_transitions = utils.read_samples(C.workspace + "/" + C.path_samples)

    tm = TransferModule(autoencoders, loss=F.l1_loss)
    dqn.reset()
    tm.reset()
    utils.set_seed(seed=C.seed, env=env)
    env.seed(C.seed)
    rrr=[]
    for n in range(N):
        s = env.reset()
        done = False
        rr=0
        while not done:
            env.render()
            a = dqn.pi(s, np.zeros(env.action_space.n))
            s_, r_, done, info = env.step(a)
            rr+=r_
            t = (s.tolist(), a, r_, s_.tolist())
            tm.push(*t)
            dqn.update(*t)
            s = s_
        rrr.append(rr)
        # if len(rrr)%5==0 and len(rrr)>0:
        #     print("n",n)
        #     xxx = np.mean(np.reshape(np.array(rrr),(int(len(rrr)/5),-1)),axis=1)
        #     plt.plot(range(len(xxx)),xxx)
        #     plt.show()


if __name__ == "__main__":
    # execute only if run as a script
    main()
