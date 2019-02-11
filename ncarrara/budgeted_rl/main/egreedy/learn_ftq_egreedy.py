# coding=utf-8
from ncarrara.budgeted_rl.tools.features import feature_factory
from ncarrara.utils import math_utils
from ncarrara.utils.math_utils import set_seed
from ncarrara.utils.os import makedirs
from ncarrara.utils_rl.algorithms.pytorch_fittedq import NetFTQ, PytorchFittedQ
from ncarrara.utils_rl.environments import envs_factory
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.budgeted_rl.tools.policies import PytorchFittedPolicy, RandomPolicy
import ncarrara.budgeted_rl.tools.utils_run as urpy
from ncarrara.budgeted_rl.tools.policies import EpsilonGreedyPolicy
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(generate_envs, feature_str, gamma, gamma_c, ftq_params, ftq_net_params,
         device, epsilon_decay, N_trajs, trajs_by_ftq_batch, normalize_reward,
         workspace, seed,lambda_=0, **args):
    envs, params = envs_factory.generate_envs(**generate_envs)
    e = envs[0]
    set_seed(seed, e)

    feature = feature_factory(feature_str)

    ftq = PytorchFittedQ(
        device=device,
        policy_network=NetFTQ(n_in=len(feature(e.reset(), e)), n_out=e.action_space.n, **ftq_net_params),
        action_str=None if not hasattr(e, "action_str") else e.action_str,
        test_policy=None,
        gamma=gamma,
        **ftq_params
    )

    decays = math_utils.epsilon_decay(**epsilon_decay, N=N_trajs, savepath=workspace)
    pi_greedy = RandomPolicy()
    pi_epsilon_greedy = EpsilonGreedyPolicy(pi_greedy=pi_greedy, epsilon=decays[0], pi_random=RandomPolicy())
    rez = np.zeros((N_trajs, 4))
    rm = Memory()
    batch = 0

    for i_traj in range(N_trajs):
        if i_traj % 50 == 0: logger.info(i_traj)
        pi_epsilon_greedy.epsilon = decays[i_traj]
        pi_epsilon_greedy.pi_greedy = pi_greedy
        trajectory, rew_r, rew_c, ret_r, ret_c = urpy.execute_policy_one_trajectory(
            e, pi_epsilon_greedy, gamma_r=gamma, gamma_c=gamma_c)
        rez[i_traj] = np.array([rew_r, rew_c, ret_r, ret_c])
        for sample in trajectory:
            rm.push(*sample)
        if i_traj > 0 and (i_traj + 1) % trajs_by_ftq_batch == 0:
            transitions_ftq, _ = urpy.datas_to_transitions(
                rm.memory, e, feature, lambda_, normalize_reward)
            logger.info("[BATCH={}]---------------------------------------".format(batch))
            logger.info("[BATCH={}][learning ftq pi greedy] #samples={} #traj={}"
                        .format(batch, len(transitions_ftq), i_traj + 1))
            logger.info("[BATCH={}]---------------------------------------".format(batch))
            ftq.reset(True)
            ftq.workspace = workspace + "/batch={}".format(batch)
            makedirs(ftq.workspace)
            pi = ftq.fit(transitions_ftq)
            ftq.save_policy()
            os.system("cp {}/policy.pt {}/final_policy.pt".format(ftq.workspace, workspace))
            pi_greedy = PytorchFittedPolicy(pi, e, feature)
            batch += 1


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        config_file = sys.argv[1]
        lambda_ = float(sys.argv[2])
        force=False
    else:
        config_file = "../config/test_egreedy.json"
        lambda_ = 0.
        force=True
    from ncarrara.budgeted_rl.tools.configuration import C
    C.load(config_file).create_fresh_workspace(force=force).load_pytorch().load_matplotlib('agg')
    main(lambda_=lambda_,
         seed = C.seed,
         device=C.device,
         workspace=C.path_learn_ftq_egreedy,
         **C.dict["learn_ftq_egreedy"],
         **C.dict)
