# coding=utf-8
import matplotlib
from bftq_pydial.tools.features import feature_0

matplotlib.use("template")
from bftq_pydial.tools.policies import PytorchFittedPolicy, PytorchBudgetedFittedPolicy
import utils_rl.algorithms.pytorch_fittedq as pftq
import bftq_pydial.bftq.pytorch_budgeted_fittedq as pbftq
import numpy as np
from gym_pydial.env.env_pydial import EnvPydial
import bftq_pydial.tools.utils_run_pydial as urpy
from bftq_pydial.tools.configuration import C


def main():
    e = EnvPydial(config_file=C.pydial_configuration, error_rate=0.3)
    print(e.cl)
    for k, v in e.cl.items():
        v.setLevel("ERROR")
    e.reset()
    action_space = e.action_space()
    feature = feature_0

    size_state = len(feature(e.reset(), e))
    print("neural net input size :", size_state)

    policy_network = pftq.NetFTQ(n_in=size_state,
                                 n_out=len(action_space),
                                 **C["main"]["net_params"])

    policy_network_bftq = pbftq.NetBFTQ(size_state=size_state,
                                        layers=C["main"]["net_params"]["intra_layers"] + [2 * len(action_space)],
                                        **C["main"]["net_params"], **C["main"]["net_bftq_params"])
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
        N_actions=len(e.action_space()),
        actions_str=e.action_space_str(),
        policy_network=policy_network_bftq,
        **C["main"]["ftq_params"],
        **C["main"]["bftq_params"]
    )

    ftq = pftq.PytorchFittedQ(
        action_str=e.action_space_str(),
        policy_network=policy_network,
        **C["main"]["ftq_params"]
    )
    datas = urpy.load_datas(C["main"]["path_sample_data"])
    transitions_ftq, transition_bftq = urpy.datas_to_transitions(datas, e, feature,
                                                                 C["main"]["lambda_"],
                                                                 C["main"]["normalize_reward"])

    bftq.reset(C["main"]["reset_weight"])
    pi_bftq = bftq.fit(transition_bftq)
    pi_bftq = PytorchBudgetedFittedPolicy(pi_bftq, e.action_space(), e, feature)
    _, results_bftq = urpy.execute_policy(e, pi_bftq,
                                          C["main"]["ftq_params"]["gamma"],
                                          C["main"]["bftq_params"]["gamma_c"],
                                          N_dialogues=C["main"]["N_trajs"],
                                          print_dial=False)

    print(np.mean(results_bftq, axis=1))

    ftq.reset(C["main"]["reset_weight"])
    pi = ftq.fit(transitions_ftq)
    pi = PytorchFittedPolicy(pi, e.action_space(), e, feature)
    _, results = urpy.execute_policy(e, pi,
                                     C["main"]["ftq_params"]["gamma"],
                                     C["main"]["bftq_params"]["gamma_c"],
                                     N_dialogues=C["main"]["N_trajs"],
                                     print_dial=False)
    urpy.print_results(results)
if __name__=="__main__":
    C.load("config_main_pydial/test.json")
    main()
