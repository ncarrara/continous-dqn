from ncarrara.continuous_dqn.dqn.transfer_module import TransferModule
import torch.nn.functional as F
from ncarrara.continuous_dqn.tools import utils
import random
import numpy as np
import logging

from ncarrara.utils.math_utils import epsilon_decay, set_seed
from ncarrara.utils_rl.algorithms.dqn import NetDQN, DQN

logger = logging.getLogger(__name__)


# def run_dqn_with_transfer(env, workspace,device, autoencoders, ers, net_params, dqn_params, decay,
#                           N, seed, test_params,
#                           sources_params, traj_max_size, feature_autoencoder, feature_dqn, start_decay):
#     return run_dqn(env, device, autoencoders, ers, workspace=workspace,net_params=net_params, dqn_params=dqn_params, decay=decay, N=N,
#                    seed=seed, test_params=test_params, sources_params=sources_params, traj_max_size=traj_max_size,
#                    feature_autoencoder=feature_autoencoder, feature_dqn=feature_dqn, start_decay=start_decay)
#
#
# def run_dqn_without_transfer(env,workspace, device, net_params, dqn_params,
#                              decay, N, seed, traj_max_size,
#                              feature_dqn, start_decay):
#     return run_dqn(env, device, workspace=workspace,autoencoders=None, ers=None,
#                    net_params=net_params, dqn_params=dqn_params, decay=decay,
#                    N=N, seed=seed, test_params=None, sources_params=None, traj_max_size=traj_max_size,
#                    feature_dqn=feature_dqn, feature_autoencoder=None, start_decay=start_decay)


def run_dqn(env, workspace, device, net_params, dqn_params, decay, N, seed, traj_max_size, feature_dqn, start_decay,
            transfer_params=None):
    # transfer_params = {autoencoders, ers,feature_autoencoder,sources_params,test_params}

    if transfer_params is not None:
        tm = TransferModule(models=transfer_params["autoencoders"],
                            loss=F.l1_loss,
                            experience_replays=transfer_params["ers"])
        tm.reset()
    size_state = len(feature_dqn(env.reset()))
    net = NetDQN(n_in=size_state, n_out=env.action_space.n, **net_params)
    dqn = DQN(policy_network=net, device=device, workspace=workspace, **dqn_params)
    dqn.reset()
    set_seed(seed=seed, env=env)
    rrr = []
    rrr_greedy = []
    epsilons = epsilon_decay(start=start_decay, decay=decay, N=N, savepath=workspace)
    nb_samples = 0
    already_warn_for_no_need_transfer = False
    for n in range(N):
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
                    a = dqn.pi(feature_dqn(s), action_mask)
                else:
                    a = dqn.pi(feature_dqn(s), np.zeros(env.action_space.n))

            s_, r_, done, info = env.step(a)
            rr += r_
            t_dqn = (feature_dqn(s), a, r_, feature_dqn(s_), done, info)
            dqn.update(*t_dqn)
            if transfer_params is not None:
                # if replay memory mini batch full, no need to transfer anymore
                if dqn.no_need_for_transfer_anymore:
                    if not already_warn_for_no_need_transfer:
                        logger.warning("No need to transfer anymore (number of trajs = {}".format(n))
                        already_warn_for_no_need_transfer = True
                else:
                    t_autoencoder = transfer_params["feature_autoencoder"]((s, a, r_, s_, done, info))
                    tm.push(t_autoencoder)
                    best_er, errors = tm.best_source_transitions()
                    dqn.update_transfer_experience_replay(best_er)
                    if transfer_params and nb_samples < 10:
                        logger.info("[N_trajs={},N_samples={}] {}".format(n, nb_samples,
                                                                          utils.format_errors(errors,
                                                                                              transfer_params[
                                                                                                  "sources_params"],
                                                                                              transfer_params[
                                                                                                  "test_params"])))

            s = s_
            nb_samples += 1
            it += 1
            if it % 100 == 0:
                if it > 500:
                    logger.warning("Number of trajectories overflowing : {}".format(it))
                # else:
                #     logger.info("step={}".format(it))
            if traj_max_size is not None and it >= traj_max_size:
                logger.warning("Max size trajectory reached")
                break
        if transfer_params is not None and n % 50 == 0:
            logger.info("------------------------------------")
            logger.info("[N_trajs={},N_samples={}] {}".format(n, nb_samples, utils.format_errors(errors,
                                                                                                 transfer_params[
                                                                                                     "sources_params"],
                                                                                                 transfer_params[
                                                                                                     "test_params"])))
            logger.info("--------------------------------------")
        # print("------------------------------->",it,rr)
        rrr.append(rr)

        s = env.reset()
        done = False
        rr = 0
        it = 0
        while not done:
            if hasattr(env, "action_space_executable"):
                exec = env.action_space_executable()
                action_mask = np.ones(env.action_space.n)
                for ex in exec:
                    action_mask[ex] = 0.
                a = dqn.pi(feature_dqn(s), action_mask)
            else:
                a = dqn.pi(feature_dqn(s), np.zeros(env.action_space.n))

            s_, r_, done, info = env.step(a)
            rr += r_
            s = s_
            it += 1
            if it % 100 == 0:
                if it > 500:
                    logger.warning("Number of trajectories overflowing : {}".format(it))
                # else:
                #     logger.info("step={}".format(it))
            if traj_max_size is not None and it >= traj_max_size:
                logger.warning("Max size trajectory reached")
                break
        rrr_greedy.append(rr)
    return rrr, rrr_greedy
