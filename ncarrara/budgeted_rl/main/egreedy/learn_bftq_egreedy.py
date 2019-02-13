# coding=utf-8
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt

from ncarrara.budgeted_rl.bftq.pytorch_budgeted_fittedq import PytorchBudgetedFittedQ, NetBFTQ
from ncarrara.budgeted_rl.tools.features import feature_factory
from ncarrara.budgeted_rl.tools.utils_run import execute_policy_from_config, datas_to_transitions
from ncarrara.utils import math_utils
from ncarrara.utils.math_utils import set_seed, near_split, zip_with_singletons
from ncarrara.utils.os import makedirs
from ncarrara.utils.torch_utils import get_memory_for_pid
from ncarrara.utils_rl.environments import envs_factory
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.budgeted_rl.tools.policies import RandomBudgetedPolicy, PytorchBudgetedFittedPolicy
from ncarrara.budgeted_rl.tools.policies import EpsilonGreedyPolicy

import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(generate_envs, feature_str, betas_for_exploration, gamma, gamma_c, bftq_params, bftq_net_params, N_trajs,
         workspace, seed, device, normalize_reward, trajs_by_ftq_batch, epsilon_decay, **args):
    # Prepare BFTQ
    envs, params = envs_factory.generate_envs(**generate_envs)
    e = envs[0]
    set_seed(seed, e)
    rm = Memory()
    feature = feature_factory(feature_str)

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

    # Prepare learning
    decays = math_utils.epsilon_decay(**epsilon_decay, N=N_trajs, savepath=workspace)
    betas_for_exploration = eval(betas_for_exploration)
    n_workers = 2
    i_traj = 0
    memory_by_batch = [get_current_memory()]
    batch_sizes = near_split(N_trajs, size_bins=trajs_by_ftq_batch)
    pi_epsilon_greedy_config = {
        "__class__": repr(EpsilonGreedyPolicy),
        "pi_greedy": {"__class__": repr(RandomBudgetedPolicy)},
        "pi_random": {"__class__": repr(RandomBudgetedPolicy)},
        "epsilon": decays[0]
    }

    # Main loop
    for batch, batch_size in enumerate(batch_sizes):
        # Prepare workers
        workers_n_trajectories = near_split(batch_size, n_workers)
        workers_start = np.cumsum(workers_n_trajectories)
        workers_traj_indexes = [np.arange(*times) for times in zip(np.insert(workers_start[:-1], 0, 0), workers_start)]
        if betas_for_exploration.size:
            workers_betas = [betas_for_exploration.take(indexes, mode='wrap') for indexes in workers_traj_indexes]
        else:
            workers_betas = [np.random.random(indexes.size) for indexes in workers_traj_indexes]
        workers_seeds = list(range(seed, seed + n_workers))
        workers_epsilons = [decays[i_traj + indexes] for indexes in workers_traj_indexes]
        workers_params = list(zip_with_singletons(generate_envs,
                                                  pi_epsilon_greedy_config,
                                                  workers_seeds,
                                                  gamma,
                                                  gamma_c,
                                                  workers_n_trajectories,
                                                  workers_betas,
                                                  workers_epsilons,
                                                  None,
                                                  args["general"]["dictConfig"]))
        # Collect trajectories
        with Pool(n_workers) as p:
            results = p.starmap(execute_policy_from_config, workers_params)
        i_traj += sum([len(trajectories) for trajectories, _ in results])

        # Fill memory
        [rm.push(*sample) for trajectories, _ in results for trajectory in trajectories for sample in trajectory]
        transitions_ftq, transition_bftq = datas_to_transitions(rm.memory, e, feature, 0, normalize_reward)

        # Fit model
        logger.info("[BATCH={}]---------------------------------------".format(batch))
        logger.info("[BATCH={}][learning bftq pi greedy] #samples={} #traj={}"
                    .format(batch, len(transitions_ftq), i_traj + 1))
        logger.info("[BATCH={}]---------------------------------------".format(batch))
        bftq = build_fresh_bftq()
        bftq.reset(True)
        bftq.workspace = workspace + "/batch={}".format(batch)
        makedirs(bftq.workspace)
        bftq.fit(transition_bftq)

        # Save policy
        network_path = bftq.save_policy()
        os.system("cp {}/policy.pt {}/final_policy.pt".format(bftq.workspace, workspace))

        # Save memory
        save_memory(bftq, memory_by_batch, by_batch=False)

        # Update greedy policy
        pi_epsilon_greedy_config["pi_greedy"] = {
            "__class__": repr(PytorchBudgetedFittedPolicy),
            "feature_str": feature_str,
            "network_path": network_path,
            "betas_for_discretisation": bftq.betas_for_discretisation,
            "device": bftq.device
        }

    save_memory(bftq, memory_by_batch, by_batch=True)


def save_memory(bftq, memory_by_batch, by_batch=False):
    if not by_batch:
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
    else:
        plt.plot(range(len(memory_by_batch)), memory_by_batch)
        plt.grid()
        plt.title("memory_by_batch")
        plt.savefig(bftq.workspace + "/memory_by_batch.png")
        plt.close()


def get_current_memory():
    return sum(get_memory_for_pid(os.getpid()))


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
