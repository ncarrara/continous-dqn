# coding=utf-8
import matplotlib

from ncarrara.bftq_pydial.tools.configuration import C
from ncarrara.bftq_pydial.tools.features import feature_0
from ncarrara.utils_rl.algorithms.pytorch_fittedq import NetFTQ, PytorchFittedQ
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.bftq_pydial.tools.policies import PytorchFittedPolicy, PytorchBudgetedFittedPolicy
from ncarrara import utils_rl as pftq
import ncarrara.bftq_pydial.bftq.pytorch_budgeted_fittedq as pbftq
import numpy as np
from gym_pydial.env.env_pydial import EnvPydial
import ncarrara.bftq_pydial.tools.utils_run_pydial as urpy


def main():
    e = EnvPydial(config_file=C.pydial_configuration,
                  seed=C["general"]["seed"],
                  error_rate=0.3,
                  pydial_logging_level="ERROR")
    e.reset()
    action_space = e.action_space
    feature = feature_0

    size_state = len(feature(e.reset(), e))
    print("neural net input size :", size_state)

    policy_network = NetFTQ(n_in=size_state,
                                 n_out=action_space.n,
                                 **C["net_params"])

    policy_network_bftq = pbftq.NetBFTQ(size_state=size_state,
                                        layers=C["net_params"]["intra_layers"] + [2 * action_space.n],
                                        **C["net_params"], **C["net_bftq_params"])
    type_beta = "EXPONENTIAL"
    nb_betas = 10
    if type_beta == "EXPONENTIAL":
        betas = np.concatenate(
            (
                np.array([0.]), np.exp(np.power(np.linspace(0, nb_betas, nb_betas), np.full(nb_betas, 2. / 3.))) / (
                    np.exp(np.power(nb_betas, 2. / 3.)))))
    elif type_beta == "LINSPACE":
        betas = np.linspace(0, 1, nb_betas + 1)
    else:
        raise Exception("type_beta inconnu : " + str(type_beta))

    bftq = pbftq.PytorchBudgetedFittedQ(
        betas=betas,
        betas_for_discretisation=betas,
        N_actions=e.action_space.n,
        actions_str=e.action_space_str,
        policy_network=policy_network_bftq,
        **C["ftq_params"],
        **C["bftq_params"]
    )

    ftq = PytorchFittedQ(
        action_str=e.action_space_str,
        policy_network=policy_network,
        **C["ftq_params"]
    )
    rm = Memory()
    rm.load_memory(C["main"]["path_sample_data"])
    transitions_ftq, transition_bftq = urpy.datas_to_transitions(rm.memory, e, feature,
                                                                 C["main"]["lambda_"],
                                                                 C["main"]["normalize_reward"])

    bftq.reset(C["main"]["reset_weight"])
    pi_bftq = bftq.fit(transition_bftq)
    pi_bftq = PytorchBudgetedFittedPolicy(pi_bftq, e.action_space, e, feature)
    _, results_bftq = urpy.execute_policy(e, pi_bftq,
                                          C["ftq_params"]["gamma"],
                                          C["bftq_params"]["gamma_c"],
                                          N_dialogues=C["main"]["N_trajs"],
                                          print_dial=False)

    urpy.print_results(results_bftq)

    ftq.reset(C["main"]["reset_weight"])
    pi = ftq.fit(transitions_ftq)
    pi = PytorchFittedPolicy(pi, e.action_space, e, feature)
    _, results = urpy.execute_policy(e, pi,
                                     C["ftq_params"]["gamma"],
                                     C["bftq_params"]["gamma_c"],
                                     N_dialogues=C["main"]["N_trajs"],
                                     print_dial=False)
    urpy.print_results(results)


if __name__ == "__main__":
    C.load("config_main_pydial/test.json")
    main()
