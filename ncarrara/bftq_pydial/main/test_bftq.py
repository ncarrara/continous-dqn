# coding=utf-8
from ncarrara.bftq_pydial.bftq.pytorch_budgeted_fittedq import NetBFTQ, PytorchBudgetedFittedQ
from ncarrara.bftq_pydial.tools.configuration import C
from ncarrara.bftq_pydial.tools.features import feature_factory
from ncarrara.utils.math import set_seed
from ncarrara.utils.os import empty_directory, makedirs
from ncarrara.utils_rl.environments.envs_factory import generate_envs
from ncarrara.bftq_pydial.tools.policies import PytorchBudgetedFittedPolicy
import ncarrara.bftq_pydial.tools.utils_run_pydial as urpy
import logging
import matplotlib.pyplot as plt
import numpy as np


def main(betas_test=None,policy_path=None,policy_basename=None):
    envs, params = generate_envs(**C["generate_envs"])
    e = envs[0]
    e.reset()
    feature = feature_factory(C["feature_str"])

    policy_network_bftq = NetBFTQ(size_state=len(feature(e.reset(), e)),
                                  layers=C["bftq_net_params"]["intra_layers"] + [2 * e.action_space.n],
                                  **C["bftq_net_params"])

    if betas_test is None:
        betas_test = eval(C["test_bftq"]["betas_test"])
    action_str = getattr(e, "action_space_str", map(str, range(e.action_space.n)))

    bftq = PytorchBudgetedFittedQ(
        device=C.device,
        workspace=C.path_bftq,
        N_actions=e.action_space.n,
        actions_str=action_str,
        policy_network=policy_network_bftq,
        gamma=C["gamma"],
        gamma_c=C["gamma_c"],
        **C["bftq_params"],

    )

    if policy_path is None:
        policy_path = C.path_bftq
    if policy_basename is None:
        policy_basename="policy"

    pi_bftq = bftq.load_policy(policy_path=policy_path,policy_basename=policy_basename)

    pi_bftq = PytorchBudgetedFittedPolicy(pi_bftq, e, feature)

    makedirs(C.path_bftq_results)
    for beta in betas_test:
        set_seed(C.seed, e)
        _, results_bftq = urpy.execute_policy(e, pi_bftq,
                                              C["gamma"],
                                              C["gamma_c"],
                                              N_dialogues=C["test_bftq"]["N_trajs"],
                                              save_path="{}/beta={}.results".format(C.path_bftq_results, beta),
                                              beta=beta)

        print("BFTQ({}) : {}".format(beta, urpy.format_results(results_bftq)))
