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


def main(betas_test, empty_previous_test=False):
    logger = logging.getLogger(__name__)
    if empty_previous_test:
        empty_directory(C.path_bftq_results)
    envs, params = generate_envs(**C["generate_envs"])
    e = envs[0]
    e.reset()
    feature = feature_factory(C["feature_str"])

    size_state = len(feature(e.reset(), e))
    print("neural net input size :", size_state)

    policy_network_bftq = NetBFTQ(size_state=size_state,
                                  layers=C["bftq_net_params"]["intra_layers"] + [2 * e.action_space.n],
                                  **C["bftq_net_params"])
    type_beta = "EXPONENTIAL"
    nb_betas = C["nb_betas"]
    if type_beta == "EXPONENTIAL":
        betas = np.concatenate(
            (np.array([0.]),
             np.exp(np.power(np.linspace(0, nb_betas, nb_betas), np.full(nb_betas, 2. / 3.))) / (
                 np.exp(np.power(nb_betas, 2. / 3.)))))
        plt.plot(range(len(betas)),betas)
        plt.title("betas")
        plt.show()
        plt.close()
    elif type_beta == "LINSPACE":
        betas = np.linspace(0, 1, nb_betas + 1)
    else:
        raise Exception("type_beta inconnu : " + str(type_beta))

    bftq = PytorchBudgetedFittedQ(
        workspace=C.path_bftq,
        betas=betas,
        betas_for_discretisation=betas,
        N_actions=e.action_space.n,
        actions_str=e.action_space_str,
        policy_network=policy_network_bftq,
        **C["bftq_params"],

    )

    rm = Memory()
    rm.load_memory(C.workspace + "/" + C["main"]["filename_data"])

    transitions_ftq, transition_bftq = urpy.datas_to_transitions(rm.memory, e, feature, 0,
                                                                 C["main"]["normalize_reward"])

    bftq.reset(True)
    pi_bftq = bftq.fit(transition_bftq)
    pi_bftq = PytorchBudgetedFittedPolicy(pi_bftq, e, feature)

    for beta in betas_test:
        _, results_bftq = urpy.execute_policy(e, pi_bftq,
                                              C["gamma"],
                                              C["gamma_c"],
                                              N_dialogues=C["main"]["N_trajs"],
                                              save_path="{}/beta={}.results".format(C.path_bftq_results, beta),
                                              beta=beta)

    print("BFTQ({}) : {}".format(beta, urpy.format_results(results_bftq)))


if __name__ == "__main__":
    C.load("config/test_slot_filling.json")
    main(betas=[0, 0.5, 1.0])
