# coding=utf-8
from ncarrara.bftq_pydial.tools.configuration import C
from ncarrara.bftq_pydial.tools.features import feature_factory
from ncarrara.utils_rl.algorithms.pytorch_fittedq import NetFTQ, PytorchFittedQ
from ncarrara.utils_rl.environments.envs_factory import generate_envs
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.bftq_pydial.tools.policies import PytorchFittedPolicy, PytorchBudgetedFittedPolicy
import ncarrara.bftq_pydial.tools.utils_run_pydial as urpy
import logging


def main(lambda_):
    logger = logging.getLogger(__name__)
    logger.setLevel(C.logging_level)

    envs, params = generate_envs(**C["generate_envs"])
    e = envs[0]
    e.reset()
    feature = feature_factory(C["feature_str"])

    size_state = len(feature(e.reset(), e))
    print("neural net input size :", size_state)

    policy_network = NetFTQ(n_in=size_state,
                            n_out=e.action_space.n,
                            **C["net_params"])

    ftq = PytorchFittedQ(
        action_str=e.action_space_str,
        policy_network=policy_network,
        **C["ftq_params"]
    )
    rm = Memory()
    rm.load_memory(C["main"]["path_sample_data"])
    transitions_ftq, transition_bftq = urpy.datas_to_transitions(rm.memory, e, feature,
                                                                 lambda_,
                                                                 C["main"]["normalize_reward"])

    ftq.reset(C["main"]["reset_weight"])
    pi = ftq.fit(transitions_ftq)
    pi = PytorchFittedPolicy(pi, e, feature)
    _, results = urpy.execute_policy(e, pi,
                                     C["gamma"],
                                     C["gamma_c"],
                                     N_dialogues=C["main"]["N_trajs"])
    urpy.print_results(results)

    # policy_network_bftq = pbftq.NetBFTQ(size_state=size_state,
    #                                     layers=C["net_params"]["intra_layers"] + [2 * action_space.n],
    #                                     **C["net_params"], **C["net_bftq_params"])
    # type_beta = "EXPONENTIAL"
    # nb_betas = 10
    # if type_beta == "EXPONENTIAL":
    #     betas = np.concatenate(
    #         (
    #             np.array([0.]), np.exp(np.power(np.linspace(0, nb_betas, nb_betas), np.full(nb_betas, 2. / 3.))) / (
    #                 np.exp(np.power(nb_betas, 2. / 3.)))))
    # elif type_beta == "LINSPACE":
    #     betas = np.linspace(0, 1, nb_betas + 1)
    # else:
    #     raise Exception("type_beta inconnu : " + str(type_beta))

    # bftq = pbftq.PytorchBudgetedFittedQ(
    #     betas=betas,
    #     betas_for_discretisation=betas,
    #     N_actions=e.action_space.n,
    #     actions_str=e.action_space_str,
    #     policy_network=policy_network_bftq,
    #     **C["ftq_params"],
    #     **C["bftq_params"]
    # )

    # bftq.reset(C["main"]["reset_weight"])
    # pi_bftq = bftq.fit(transition_bftq)
    # pi_bftq = PytorchBudgetedFittedPolicy(pi_bftq, e.action_space, e, feature)
    # _, results_bftq = urpy.execute_policy(e, pi_bftq,
    #                                       C["ftq_params"]["gamma"],
    #                                       C["bftq_params"]["gamma_c"],
    #                                       N_dialogues=C["main"]["N_trajs"],
    #                                       print_dial=False)
    #
    # urpy.print_results(results_bftq)


if __name__ == "__main__":
    C.load("config/test_slot_filling.json")
    main(lambda_=0)
