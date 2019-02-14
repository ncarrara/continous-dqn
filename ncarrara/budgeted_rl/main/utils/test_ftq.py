# coding=utf-8
import os
from multiprocessing.pool import Pool

import numpy as np

from ncarrara.budgeted_rl.tools.utils_run import execute_policy_from_config, format_results
from ncarrara.utils.math_utils import set_seed, near_split, zip_with_singletons
from ncarrara.utils.os import makedirs
from ncarrara.budgeted_rl.tools.policies import PytorchFittedPolicy

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(device, workspace, policy_path, generate_envs, feature_str, gamma, gamma_c,
         N_trajs, seed, lambda_, path_results, general, **args):
    if not os.path.isabs(policy_path):
        policy_path = workspace + "/" + policy_path

    pi_config = {
        "__class__": repr(PytorchFittedPolicy),
        "feature_str": feature_str,
        "network_path": policy_path,
        "device": device
    }

    makedirs(path_results)
    set_seed(seed)
    # Prepare workers
    cpu_processes = min(general["cpu"]["processes_when_linked_with_gpu"] or os.cpu_count(), N_trajs)
    workers_n_trajectories = near_split(N_trajs, cpu_processes)
    workers_seeds = np.random.randint(0, 10000, cpu_processes).tolist()
    workers_params = list(zip_with_singletons(
        generate_envs, pi_config, workers_seeds, gamma, gamma_c, workers_n_trajectories, None,
        None, "{}/lambda={}.results".format(path_results, lambda_), general["dictConfig"]))
    # Collect trajectories
    logger.info("Collecting trajectories with {} workers...".format(cpu_processes))
    with Pool(cpu_processes) as pool:
        results = pool.starmap(execute_policy_from_config, workers_params)
        results = np.concatenate([result for _, result in results], axis=0)

    print("FTQ({}) : {}".format(lambda_, format_results(results)))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "../config/test_egreedy.json"
    from ncarrara.budgeted_rl.tools.configuration import C

    C.load(config_file).load_pytorch().load_matplotlib('agg')
    main(lambda_=0,
         device=C.device,
         seed=C.seed,
         workspace=C.path_learn_ftq_egreedy,
         path_results=C.path_learn_ftq_egreedy,
         **C.dict["test_ftq"],
         **C.dict)
