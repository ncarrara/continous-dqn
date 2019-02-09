# coding=utf-8
from ncarrara.budgeted_rl.bftq.pytorch_budgeted_fittedq import PytorchBudgetedFittedQ, NetBFTQ
from ncarrara.budgeted_rl.tools.features import feature_factory
from ncarrara.utils import math
from ncarrara.utils.math import  set_seed
from ncarrara.utils.os import makedirs
from ncarrara.utils_rl.environments import envs_factory
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.budgeted_rl.tools.policies import RandomBudgetedPolicy, PytorchBudgetedFittedPolicy
import ncarrara.budgeted_rl.tools.utils_run as urpy
from ncarrara.budgeted_rl.tools.policies import EpsilonGreedyPolicy

import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(generate_envs, feature_str, betas_for_exploration, gamma, gamma_c, bftq_params, bftq_net_params, N_trajs,
         workspace, seed, device, normalize_reward, trajs_by_ftq_batch,epsilon_decay,**args):
    envs, params = envs_factory.generate_envs(**generate_envs)
    e = envs[0]
    set_seed(seed, e)

    feature = feature_factory(feature_str)

    betas_for_exploration = eval(betas_for_exploration)

    bftq = PytorchBudgetedFittedQ(
        device=device,
        workspace=workspace + "/batch=0",
        actions_str=None if not hasattr( e,"action_str") else e.action_str,
        policy_network=NetBFTQ(size_state=len(feature(e.reset(), e)), n_actions=e.action_space.n, **bftq_net_params),
        gamma=gamma,
        gamma_c=gamma_c,
        **bftq_params

    )

    pi_greedy = None

    decays = math.epsilon_decay(**epsilon_decay, N=N_trajs, savepath=workspace)

    pi_random = RandomBudgetedPolicy()
    pi_epsilon_greedy = EpsilonGreedyPolicy(pi_greedy, decays[0], pi_random=pi_random)
    pi_greedy = pi_random
    rez = np.zeros((N_trajs, 4))
    rm = Memory()
    batch = 0

    for i_traj in range(N_trajs):
        if i_traj % 50 == 0: logger.info(i_traj)
        pi_epsilon_greedy.epsilon = decays[i_traj]
        pi_epsilon_greedy.pi_greedy = pi_greedy
        if len(betas_for_exploration) == 0:
            init_betas = [np.random.sample()]
        else:
            init_betas = betas_for_exploration
        for beta in init_betas:
            trajectory, rew_r, rew_c, ret_r, ret_c = urpy.execute_policy_one_trajectory(
                e, pi_epsilon_greedy, gamma_r=gamma, gamma_c=gamma_c, beta=beta)
            rez[i_traj] = np.array([rew_r, rew_c, ret_r, ret_c])
            for sample in trajectory:
                rm.push(*sample)
        if i_traj > 0 and (i_traj + 1) % trajs_by_ftq_batch == 0:
            transitions_ftq, transition_bftq = urpy.datas_to_transitions(
                rm.memory, e, feature, 0, normalize_reward)
            logger.info("[BATCH={}]---------------------------------------".format(batch))
            logger.info("[BATCH={}][learning bftq pi greedy] #samples={} #traj={}"
                        .format(batch, len(transitions_ftq), i_traj + 1))
            logger.info("[BATCH={}]---------------------------------------".format(batch))
            bftq.reset(True)
            bftq.workspace = workspace + "/batch={}".format(batch)
            makedirs(bftq.workspace)
            pi = bftq.fit(transition_bftq)
            bftq.save_policy()
            os.system("cp {}/policy.pt {}/final_policy.pt".format(bftq.workspace, workspace))
            pi_greedy = PytorchBudgetedFittedPolicy(pi, e, feature)
            batch += 1


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        force=False
    else:
        config_file = "../config/test_egreedy.json"
        force=True
    from ncarrara.budgeted_rl.tools.configuration import C
    C.load(config_file).create_fresh_workspace(force=force).load_pytorch().load_matplotlib('agg')
    main(device=C.device,
         seed=C.seed,
         workspace=C.path_learn_bftq_egreedy,
         **C.dict["learn_bftq_egreedy"],
         **C.dict)
