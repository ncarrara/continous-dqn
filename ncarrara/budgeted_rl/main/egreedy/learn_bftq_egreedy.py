# coding=utf-8
from ncarrara.budgeted_rl.bftq.pytorch_budgeted_fittedq import PytorchBudgetedFittedQ, NetBFTQ
from ncarrara.budgeted_rl.tools.features import feature_factory
from ncarrara.utils import math_utils
from ncarrara.utils.math_utils import set_seed
from ncarrara.utils.os import makedirs
from ncarrara.utils.torch_utils import get_memory_for_pid
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


def get_current_memory():
    memory = get_memory_for_pid(os.getpid())
    sum = 0
    for mem in memory:
        sum += mem

    return sum


def main(generate_envs, feature_str, betas_for_exploration, gamma, gamma_c, bftq_params, bftq_net_params, N_trajs,
         workspace, seed, device, normalize_reward, trajs_by_ftq_batch, epsilon_decay, **args):
    envs, params = envs_factory.generate_envs(**generate_envs)
    e = envs[0]
    set_seed(seed, e)

    feature = feature_factory(feature_str)

    betas_for_exploration = eval(betas_for_exploration)

    def build_fresh_bftq():
        bftq = PytorchBudgetedFittedQ(
            device=device,
            workspace=workspace + "/batch=0",
            actions_str=None if not hasattr(e, "action_str") else e.action_str,
            policy_network=NetBFTQ(size_state=len(feature(e.reset(), e)), n_actions=e.action_space.n,
                                   **bftq_net_params),
            gamma=gamma,
            gamma_c=gamma_c,
            **bftq_params)
        return bftq

    pi_greedy = None

    decays = math_utils.epsilon_decay(**epsilon_decay, N=N_trajs, savepath=workspace)

    pi_random = RandomBudgetedPolicy()
    pi_epsilon_greedy = EpsilonGreedyPolicy(pi_greedy, decays[0], pi_random=pi_random)
    pi_greedy = pi_random
    rez = np.zeros((N_trajs, 4))
    rm = Memory()
    batch = 0
    i_traj = 0
    # for i_traj in range(N_trajs):
    memory_by_batch = [get_current_memory()]
    while i_traj < N_trajs:
        if len(betas_for_exploration) == 0:
            init_betas = [np.random.sample()]
        else:
            init_betas = betas_for_exploration
        for beta in init_betas:
            if i_traj % 50 == 0: logger.info(i_traj)
            pi_epsilon_greedy.epsilon = decays[i_traj]
            pi_epsilon_greedy.pi_greedy = pi_greedy
            trajectory, rew_r, rew_c, ret_r, ret_c = urpy.execute_policy_one_trajectory(
                e, pi_epsilon_greedy, gamma_r=gamma, gamma_c=gamma_c, beta=beta)
            rez[i_traj] = np.array([rew_r, rew_c, ret_r, ret_c])
            for sample in trajectory:
                rm.push(*sample)
            if (i_traj + 1) % trajs_by_ftq_batch == 0:
                transitions_ftq, transition_bftq = urpy.datas_to_transitions(
                    rm.memory, e, feature, 0, normalize_reward)
                logger.info("[BATCH={}]---------------------------------------".format(batch))
                logger.info("[BATCH={}][learning bftq pi greedy] #samples={} #traj={}"
                            .format(batch, len(transitions_ftq), i_traj + 1))
                logger.info("[BATCH={}]---------------------------------------".format(batch))
                bftq = build_fresh_bftq()
                bftq.reset(True)
                bftq.workspace = workspace + "/batch={}".format(batch)
                makedirs(bftq.workspace)
                pi = bftq.fit(transition_bftq)
                bftq.save_policy()
                os.system("cp {}/policy.pt {}/final_policy.pt".format(bftq.workspace, workspace))
                pi_greedy = PytorchBudgetedFittedPolicy(pi, e, feature)
                batch += 1
                import matplotlib.pyplot as plt
                import matplotlib.pyplot as plt
                plt.rcParams["figure.figsize"] = (30, 5)
                plt.grid()
                y_mem = np.asarray(bftq.memory_tracking)[:, 1]
                y_mem = [int(mem) for mem in y_mem]
                plt.plot(range(len(y_mem)), y_mem)
                props = {'ha': 'center', 'va': 'center', 'bbox': {'fc': '0.8', 'pad': 0}}

                for i, couple in enumerate(bftq.memory_tracking):
                    id, memory = couple
                    plt.scatter(i, memory, s=25)
                    plt.text(i, memory, id, props, rotation=90)

                plt.savefig(bftq.workspace + "/memory_tracking.png")
                plt.close()
                memory_by_batch.append(get_current_memory())
            i_traj += 1
            if i_traj >= N_trajs:  # Needed because of the for-loop in betas
                break  # which continues even after N_trajs is reached
    import matplotlib.pyplot as plt

    plt.plot(range(len(memory_by_batch)), memory_by_batch)
    plt.grid()
    plt.title("memory_by_batch")
    plt.savefig(workspace + "/memory_by_batch.png")
    plt.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        config_file = sys.argv[1]
        force = bool(sys.argv[2])
    else:
        config_file = "../config/test_egreedy.json"
        force = True
    from ncarrara.budgeted_rl.tools.configuration import C

    C.load(config_file).create_fresh_workspace(force=force).load_pytorch().load_matplotlib('agg')
    main(device=C.device,
         seed=C.seed,
         workspace=C.path_learn_bftq_egreedy,
         **C.dict["learn_bftq_egreedy"],
         **C.dict)
