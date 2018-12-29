import os

from continuous_dqn.dqn.transfer_module import TransferModule
from continuous_dqn.tools.configuration import C
import torch.nn.functional as F
from continuous_dqn.tools import utils
import random
import numpy as np
import logging

from utils_rl.algorithms.dqn import DQN, NetDQN

logger = logging.getLogger(__name__)


def run_dqn_with_transfer( env, autoencoders, ers, net_params, dqn_params, decay,
                           N, seed, test_params,
                          sources_params,traj_max_size):
    return run_dqn( env, autoencoders, ers, net_params=net_params, dqn_params=dqn_params, decay=decay, N=N,
                   seed=seed, test_params=test_params, sources_params=sources_params,traj_max_size=traj_max_size)


def run_dqn_without_transfer( env, net_params, dqn_params, decay, N, seed, test_params,
                              sources_params,traj_max_size, **params):
    return run_dqn( env, autoencoders=None, ers=None,
                   net_params=net_params, dqn_params=dqn_params, decay=decay,
                   N=N, seed=seed, test_params=test_params, sources_params=sources_params,traj_max_size=traj_max_size)


def run_dqn( env, autoencoders, ers, net_params, dqn_params, decay, N, seed, test_params,
             sources_params,traj_max_size):
    do_transfer = autoencoders is not None or ers is not None
    if do_transfer:
        tm = TransferModule(models=autoencoders,
                            loss=F.l1_loss,
                            experience_replays=ers)
        tm.reset()
    net = NetDQN(**net_params)
    dqn = DQN(policy_network=net, **dqn_params)
    dqn.reset()
    utils.set_seed(seed=seed, env=env)
    env.seed(C.seed)
    # render = lambda: plt.imshow(env.render(mode='rgb_array'))

    rrr = []
    rrr_greedy = []
    epsilons = utils.epsilon_decay(start=1.0, decay=decay, N=N)

    nb_samples = 0
    for n in range(N):
        s = env.reset()
        done = False
        rr = 0
        it=0
        while (not done):

            if random.random() < epsilons[n]:
                a = env.action_space.sample()
            else:
                a = dqn.pi(s, np.zeros(env.action_space.n))
            s_, r_, done, info = env.step(a)
            rr += r_
            t = (s.tolist(), a, r_, s_.tolist(), done, info)
            dqn.update(*t)
            if do_transfer:
                tm.push(*t)

                best_er, errors = tm.best_source_transitions()

                dqn.update_transfer_experience_replay(best_er)
                if nb_samples < 10:
                    logger.info("[N_trajs={},N_samples={}] {}".format(n, nb_samples,
                                                                      utils.format_errors(errors,
                                                                                          sources_params,
                                                                                          test_params)))

            s = s_
            nb_samples += 1
            it+=1
            if traj_max_size is not None and it >= traj_max_size:
                break


        if do_transfer and n % 50 == 0:
            logger.info("------------------------------------")
            logger.info("[N_trajs={},N_samples={}] {}".format(n, nb_samples,
                                                              utils.format_errors(errors,
                                                                                  sources_params,
                                                                                  test_params)))
            logger.info("--------------------------------------")

        rrr.append(rr)

        s = env.reset()
        done = False
        rr = 0
        while not done:
            a = dqn.pi(s, np.zeros(env.action_space.n))
            s_, r_, done, info = env.step(a)
            rr += r_
            s = s_
        rrr_greedy.append(rr)
    # monitor.close()
    return rrr, rrr_greedy
    # n_dots = 10
    # if len(rrr) % int(N / n_dots) == 0 and len(rrr) > 0:
    #     print("n", n)
    #     xxx = np.mean(np.reshape(np.array(rrr), (int(len(rrr) / int(N / n_dots)), -1)), axis=1)
    #     plt.plot(range(len(xxx)), xxx)
    #     plt.show()

