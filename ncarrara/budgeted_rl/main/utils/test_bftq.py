# coding=utf-8
from ncarrara.budgeted_rl.bftq.pytorch_budgeted_fittedq import NetBFTQ, PytorchBudgetedFittedQ

from ncarrara.budgeted_rl.tools.features import feature_factory
from ncarrara.utils.math import set_seed
from ncarrara.utils.os import empty_directory, makedirs
from ncarrara.utils_rl.environments import envs_factory
from ncarrara.budgeted_rl.tools.policies import PytorchBudgetedFittedPolicy
import ncarrara.budgeted_rl.tools.utils_run as urpy
import numpy as np


def main(betas_test, policy_path, generate_envs, feature_str, device,
         workspace, gamma, gamma_c, bftq_net_params, bftq_params, seed, N_trajs, path_results, **args):
    envs, params = envs_factory.generate_envs(**generate_envs)
    e = envs[0]
    e.reset()
    feature = feature_factory(feature_str)

    betas_test = eval(betas_test)

    algo = PytorchBudgetedFittedQ(
        device=device,
        workspace=workspace,
        actions_str=None if not hasattr(e, "action_str") else e.action_str,
        policy_network=NetBFTQ(size_state=len(feature(e.reset(), e)), n_actions=e.action_space.n, **bftq_net_params),
        gamma=gamma,
        gamma_c=gamma_c,
        **bftq_params,

    )

    pi = algo.load_policy(policy_path=workspace + "/" + policy_path)

    pi = PytorchBudgetedFittedPolicy(pi, e, feature)

    makedirs(path_results)
    for beta in betas_test:
        set_seed(seed, e)
        _, results = urpy.execute_policy(env=e,
                                         pi=pi,
                                         gamma_r=gamma,
                                         gamma_c=gamma_c,
                                         N_dialogues=N_trajs,
                                         save_path="{}/beta={}.results".format(path_results, beta),
                                         beta=beta)

        print("BFTQ({}) : {}".format(beta, urpy.format_results(results)))


if __name__ == "__main__":
    import sys

    # Workspace from config and id
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "../config/test_egreedy.json"
    if len(sys.argv) > 2:
        id = sys.argv[2]
    else:
        id = None

    from ncarrara.budgeted_rl.tools.configuration import C

    C.load(config_file).load_pytorch().load_matplotlib('agg')
    if id:
        C.workspace += "/{}/".format(id)
        C.update_paths()

    main(device=C.device,
         seed=C.seed,
         workspace=C.path_learn_bftq_egreedy,
         path_results=C.path_bftq_results,
         **C.dict["test_bftq"],
         **C.dict)
