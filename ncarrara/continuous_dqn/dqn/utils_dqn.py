from ncarrara.continuous_dqn.dqn.tnn import transfer_network_factory
from ncarrara.continuous_dqn.dqn.transfer_module import TransferModule
import random
import numpy as np
import logging

from ncarrara.utils.math_utils import epsilon_decay, set_seed
from ncarrara.utils_rl.algorithms.dqn import NetDQN, DQN
from ncarrara.continuous_dqn.dqn.tdqn import TDQN
from ncarrara.utils_rl.transition.replay_memory import Memory

logger = logging.getLogger(__name__)


def run_dqn(env, workspace, device, net_params, dqn_params, decay, N, seed, feature_dqn, start_decay, gamma=None,
            transfer_params=None, evaluate_greedy_policy=True, traj_max_size=None, writer=None):
    size_state = len(feature_dqn(env.reset()))
    if transfer_params is None or transfer_params["selection_method"] == "no_transfer":
        tm = None
    else:
        tm = TransferModule(**transfer_params)
        tm.reset()
    net = NetDQN(n_in=size_state, n_out=env.action_space.n, **net_params)
    dqn = TDQN(
        policy_network=net,
        device=device,
        transfer_module=tm,
        workspace=workspace,
        writer=writer,
        feature=feature_dqn,
        **dqn_params)
    dqn.reset()
    set_seed(seed=seed, env=env)
    rrr = []
    rrr_greedy = []
    epsilons = epsilon_decay(start=start_decay, decay=decay, N=N, savepath=workspace)
    nb_samples = 0
    memory = Memory()
    for n in range(N):
        # print("-------------------------- "+str(n)+ "----------------------")
        s = env.reset()
        done = False
        rr = 0
        it = 0
        while (not done):

            if random.random() < epsilons[n]:
                if hasattr(env, "action_space_executable"):
                    a = np.random.choice(env.action_space_executable())
                else:
                    a = env.action_space.sample()
            else:
                if hasattr(env, "action_space_executable"):
                    exec = env.action_space_executable()
                    action_mask = np.ones(env.action_space.n)
                    for ex in exec:
                        action_mask[ex] = 0.
                    a = dqn.pi(s, action_mask)
                else:
                    a = dqn.pi(s, np.zeros(env.action_space.n))

            s_, r_, done, info = env.step(a)
            done = done or (traj_max_size is not None and it >= traj_max_size - 1)
            rr += r_ * (gamma ** it)
            t_dqn = (s, a, r_, s_, done, info)
            memory.push(s, a, r_, s_, done, info)
            dqn.update(*t_dqn)
            s = s_
            nb_samples += 1
            it += 1
        if writer is not None:
            writer.add_scalar('return/episode', rr, n)
        rrr.append(rr)

        if evaluate_greedy_policy:
            s = env.reset()
            done = False
            rr_greedy = 0
            it = 0
            while (not done):
                if hasattr(env, "action_space_executable"):
                    exec = env.action_space_executable()
                    action_mask = np.ones(env.action_space.n)
                    for ex in exec:
                        action_mask[ex] = 0.
                    a = dqn.pi(s, action_mask)
                else:
                    a = dqn.pi(s, np.zeros(env.action_space.n))

                s_, r_, done, info = env.step(a)
                done = done or (traj_max_size is not None and it >= traj_max_size - 1)
                rr_greedy += r_ * (gamma ** it)
                s = s_
                it += 1
            rrr_greedy.append(rr_greedy)
            if writer is not None:
                writer.add_scalar('return_greedy/episode', rr_greedy, n)
            # print("eps={} greedy={}".format(rr,rr_greedy))
    import matplotlib.pyplot as plt
    for param_stat in ["weights_over_time", "biais_over_time",
                       "ae_errors_over_time", "p_over_time",
                       "best_fit_over_time"]:
        if hasattr(dqn, param_stat):
            var = getattr(dqn, param_stat)
            plt.plot(range(0, len(var)), var)
            plt.title(param_stat)
            plt.savefig(workspace / param_stat)
            plt.close()

    return rrr, rrr_greedy, memory, dqn

# if tm is not None and n % 50 == 0 and transfer_params["selection_method"] == "transfer":
#     logger.info("------------------------------------")
#     logger.info("[N_trajs={},N_samples={}] {}"
#                 .format(n, nb_samples, utils.format_errors(tm.errors,
#                                                            transfer_params["sources_params"],
#                                                            transfer_params["test_params"])))
#     logger.info("--------------------------------------")
