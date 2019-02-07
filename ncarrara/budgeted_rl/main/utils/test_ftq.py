# coding=utf-8
from ncarrara.budgeted_rl.bftq.pytorch_budgeted_fittedq import NetBFTQ, PytorchBudgetedFittedQ
from ncarrara.budgeted_rl.tools.configuration import C
from ncarrara.budgeted_rl.tools.features import feature_factory
from ncarrara.utils.math import set_seed
from ncarrara.utils.os import empty_directory, makedirs
from ncarrara.utils_rl.algorithms.pytorch_fittedq import NetFTQ, PytorchFittedQ
from ncarrara.budgeted_rl.tools.policies import PytorchBudgetedFittedPolicy
import ncarrara.budgeted_rl.tools.utils_run_pydial as urpy
import logging
import matplotlib.pyplot as plt
import numpy as np

from ncarrara.utils_rl.environments import envs_factory


def main(lambda_=0,
         device=C.device,
         workspace=C.path_ftq,
         policy_path=C.path_ftq,
         policy_basename="final_policy",
         path_save=C.path_ftq_results,
         generate_envs=C["generate_envs"],
         net_params=C["net_params"],
         ftq_params=C["ftq_params"],
         feature_str=C["feature_str"],
         gamma=C["gamma"],
         gamma_c=C["gamma_C"],
         N_dialogues=C["test_ftq"]["N_trajs"],
         seed=C.seed):
    envs, params = envs_factory.generate_envs(**generate_envs)
    e = envs[0]
    e.reset()
    feature = feature_factory(feature_str)

    net = NetFTQ(n_in=len(feature(e.reset(), e)),
                 n_out=e.action_space.n,
                 **net_params)

    algo = PytorchFittedQ(
        device=device,
        test_policy=None,
        workspace=workspace,
        action_str=None if not hasattr("action_str", e) else e.action_str,
        policy_network=net,
        gamma=gamma,
        **ftq_params
    )

    pi = algo.load_policy(policy_path=policy_path, policy_basename=policy_basename)

    pi = PytorchBudgetedFittedPolicy(pi, e, feature)

    makedirs(path_save)
    set_seed(seed, e)
    _, results = urpy.execute_policy(e=e,
                                     pi=pi,
                                     gamma=gamma,
                                     gamma_c=gamma_c,
                                     N_dialogues=N_dialogues,
                                     save_path="{}/lambda={}.results".format(path_save, lambda_))

    print("FTQ({}) : {}".format(lambda_, urpy.format_results(results)))
