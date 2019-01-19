# coding=utf-8
from ncarrara.bftq_pydial.bftq.pytorch_budgeted_fittedq import NetBFTQ, PytorchBudgetedFittedQ
from ncarrara.bftq_pydial.tools.configuration import C
from ncarrara.bftq_pydial.tools.features import feature_factory
from ncarrara.utils.os import empty_directory
from ncarrara.utils_rl.environments.envs_factory import generate_envs
from ncarrara.bftq_pydial.tools.policies import PytorchBudgetedFittedPolicy
import ncarrara.bftq_pydial.tools.utils_run_pydial as urpy
import logging
import matplotlib.pyplot as plt
import numpy as np

from ncarrara.utils_rl.transition.replay_memory import Memory


def main():
    logger = logging.getLogger(__name__)

    envs, params = generate_envs(**C["generate_envs"])
    e = envs[0]
    e.reset()
    feature = feature_factory(C["feature_str"])

    size_state = len(feature(e.reset(), e))
    # print("neural net input size :", size_state)

    policy_network_bftq = NetBFTQ(size_state=size_state,
                                  layers=C["bftq_net_params"]["intra_layers"] + [2 * e.action_space.n],
                                  **C["bftq_net_params"])

    betas = eval(C["betas"])

    bftq = PytorchBudgetedFittedQ(
        device=C.device,
        workspace=C.path_bftq,
        betas=betas,
        betas_for_discretisation=betas,
        N_actions=e.action_space.n,
        actions_str=e.action_space_str,
        policy_network=policy_network_bftq,
        **C["bftq_params"],

    )

    rm = Memory()
    if C["main"]["path_data"] is None:
        path_data = C.workspace + "/" + C["main"]["filename_data"]
    else:
        path_data = C["main"]["path_data"] + "/" + C["main"]["filename_data"]
    rm.load_memory(path_data)

    transitions_ftq, transition_bftq = urpy.datas_to_transitions(rm.memory, e, feature, 0,
                                                                 C["main"]["normalize_reward"])

    bftq.reset(True)
    _ = bftq.fit(transition_bftq)

    bftq.save_policy()


if __name__ == "__main__":
    C.load("config/test_slot_filling.json")
    main(betas=[0, 0.5, 1.0])
