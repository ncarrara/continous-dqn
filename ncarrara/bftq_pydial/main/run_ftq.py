# coding=utf-8
from ncarrara.bftq_pydial.tools.configuration import C
from ncarrara.bftq_pydial.tools.features import feature_factory
from ncarrara.utils.os import empty_directory, makedirs
from ncarrara.utils_rl.algorithms.pytorch_fittedq import NetFTQ, PytorchFittedQ
from ncarrara.utils_rl.environments.envs_factory import generate_envs
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.bftq_pydial.tools.policies import PytorchFittedPolicy, PytorchBudgetedFittedPolicy, \
    HandcraftedSlotFillingEnv
import ncarrara.bftq_pydial.tools.utils_run_pydial as urpy
import logging
import os
import numpy as np


def main(lambdas_, empty_previous_test=False):
    logger = logging.getLogger(__name__)
    if empty_previous_test:
        empty_directory(C.path_ftq_results)

    envs, params = generate_envs(**C["generate_envs"])
    e = envs[0]
    e.reset()
    feature =feature_factory(C["feature_str"])

    size_state = len(feature(e.reset(), e))
    logger.info("neural net input size : {}".format(size_state))

    policy_network = NetFTQ(n_in=size_state,
                            n_out=e.action_space.n,
                            **C["net_params"])

    def process_between_epoch(pi):
        logger.info("process_between_epoch ...")
        pi = PytorchFittedPolicy(pi, e, feature)
        _, results = urpy.execute_policy(e, pi, C["gamma"], C["gamma_c"], C["nb_trajs_between_epoch"], 1.)
        return np.mean(results, axis=0)

    ftq = PytorchFittedQ(
        test_policy=process_between_epoch,
        workspace=C.path_ftq,
        action_str=e.action_space_str,
        policy_network=policy_network,
        **C["ftq_params"]
    )
    rm = Memory()
    if C["main"]["path_data"] is None:
        path_data = C.workspace + "/" + C["main"]["filename_data"]
    else:
        path_data = C["main"]["path_data"] + "/" + C["main"]["filename_data"]
    rm.load_memory(path_data)
    # logging.getLogger("ncarrara.utils_rl.environments.slot_filling_env.slot_filling_env").setLevel("INFO")
    makedirs(C.path_ftq_results)
    for lambda_ in lambdas_:
        transitions_ftq, transition_bftq = urpy.datas_to_transitions(rm.memory, e, feature,
                                                                     lambda_,
                                                                     C["main"]["normalize_reward"])

        ftq.reset(True)
        pi = ftq.fit(transitions_ftq)
        pi = PytorchFittedPolicy(pi, e, feature)

        _, results = urpy.execute_policy(e, pi,
                                         C["gamma"],
                                         C["gamma_c"],
                                         N_dialogues=C["main"]["N_trajs"],
                                         save_path="{}/lambda_={}.results".format(C.path_ftq_results, lambda_))
        logger.info("FTQ({}) : {} ".format(lambda_, urpy.format_results(results)))


if __name__ == "__main__":
    C.load("config/test_slot_filling.json")
    main(lambda_=[0])

    # logging.getLogger("ncarrara.utils_rl.environments.slot_filling_env.slot_filling_env").setLevel("WARNING")

    # feats = [
    #     [1.00, 0.0, 1.00, 1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1.00, 1.00, 1.00, 1.0, 0.0, 0.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1.00, 1.00, 1.00, 1.0, 0.0, 0.0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1.00, 1.00, 1.00, 1.0, 0.0, 0.0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [1.00, 1.00, 1.00, 1.0, 0.0, 0.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [1.00, 1.00, 1.00, 1.0, 0.0, 0.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [1.00, 1.00, 1.00, 1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [1.00, 1.00, 1.00, 1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #     [1.00, 1.00, 1.00, 1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #     [1.00, 1.00, 1.00, 1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #     [1.00, 1.00, 1.00, 1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #     [0.35, 1.00, 0.80, 1.0, 0.0, 0.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [1.00, 1.00, 1.00, 1.0, 0.0, 0.0, 0, 0, 0, 1, 0, 0, 0, 0, 1., 0],
    #     [0.3, 0.1, 0.8, 1.0, 0.0, 0.0, 0, 0, 0, 1, 0, 0, 0, 0, 1., 0],
    #     [1.00, 1.00, 1.00, 1.0, 0.0, 0.0, 0, 0, 0, 1, 0, 0, 1., 0, 0, 0]
    # ]
    # for feat in feats:
    #     print(feat)
    #     q = ftq.q(feat)[0]
    #     print("".join(["{} -> {:.2f}\n".format(e.system_actions[iy], y) for iy, y in enumerate(q)]))
