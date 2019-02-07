# coding=utf-8
from ncarrara.budgeted_rl.bftq.pytorch_budgeted_fittedq import NetBFTQ, PytorchBudgetedFittedQ
from ncarrara.budgeted_rl.tools.configuration import C
from ncarrara.budgeted_rl.tools.features import feature_factory
from ncarrara.utils.math import set_seed
from ncarrara.utils.os import empty_directory, makedirs
from ncarrara.utils_rl.environments import envs_factory
from ncarrara.utils_rl.environments.envs_factory import generate_envs
from ncarrara.budgeted_rl.tools.policies import PytorchBudgetedFittedPolicy
import ncarrara.budgeted_rl.tools.utils_run_pydial as urpy
import logging
import matplotlib.pyplot as plt
import numpy as np


def main(betas_test,policy_path,policy_basename,generate_envs,feature_str,device,
         workspace,gamma,gamma_c,bftq_net_params,bftq_params,seed,N_trajs):
    envs, params = envs_factory.generate_envs(**generate_envs)
    e = envs[0]
    e.reset()
    feature = feature_factory(feature_str)



    if betas_test is None:
        betas_test = eval(betas_test)

    algo = PytorchBudgetedFittedQ(
        device=device,
        workspace=workspace,
        actions_str=None if not hasattr("action_str", e) else e.action_str,
        policy_network=NetBFTQ(size_state=len(feature(e.reset(), e)), n_actions=e.action_space.n, **bftq_net_params),
        gamma=gamma,
        gamma_c=gamma_c,
        **bftq_params,

    )

    pi = algo.load_policy(policy_path=policy_path,policy_basename=policy_basename)

    pi = PytorchBudgetedFittedPolicy(pi, e, feature)

    path_results = workspace+"/results"
    makedirs(path_results)
    for beta in betas_test:
        set_seed(seed, e)
        _, results  = urpy.execute_policy(e, pi,
                                              gamma,
                                              gamma_c,
                                              N_dialogues=N_trajs,
                                              save_path="{}/beta={}.results".format(path_results, beta),
                                              beta=beta)

        print("BFTQ({}) : {}".format(beta, urpy.format_results(results )))
