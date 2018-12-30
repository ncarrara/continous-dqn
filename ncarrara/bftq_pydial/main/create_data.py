# coding=utf-8
from ncarrara.bftq_pydial.tools.features import feature_0
from ncarrara.utils.math import epsilon_decay
from ncarrara.utils_rl.algorithms.pytorch_fittedq import NetFTQ, PytorchFittedQ
from ncarrara.utils_rl.transition.replay_memory import ReplayMemory
from ncarrara.bftq_pydial.tools.policies import PytorchFittedPolicy, RandomPolicy
import ncarrara.bftq_pydial.tools.utils_run_pydial as urpy
from ncarrara.bftq_pydial.tools.policies import EpsilonGreedyPolicy
from ncarrara.bftq_pydial.tools.configuration import C
from ncarrara.utils_rl.transition.transition import TransitionGym

import numpy as np
from gym_pydial.env.env_pydial import EnvPydial

import logging

logging.basicConfig(level=logging.INFO)


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(C.logging_level)
    e = EnvPydial(config_file=C.pydial_configuration,
                  seed=C["general"]["seed"],
                  error_rate=0.3,
                  pydial_logging_level="ERROR")
    e.reset()
    feature = feature_0

    def process_between_epoch(pi):
        logger.info("process_between_epoch ...")
        pi = PytorchFittedPolicy(pi, e.action_space, e, feature)
        _, results = urpy.execute_policy(e, pi, C["gamma"], C["gamma_c"], 5, 1., False)
        return np.mean(results, axis=0)

    size_state = len(feature(e.reset(), e))
    logger.info("neural net input size : {}".format(size_state))

    policy_network = NetFTQ(n_in=size_state, n_out=e.action_space.n, **C["net_params"])

    ftq = PytorchFittedQ(
        device=C.device,
        action_str=e.action_space_str,
        policy_network=policy_network,
        process_between_epoch=process_between_epoch, **C["ftq_params"]
    )

    pi_greedy = None

    decays = epsilon_decay(**C["create_data"]["epsilon_decay"], N=C["create_data"]["N_trajs"])

    pi_epsilon_greedy = EpsilonGreedyPolicy(pi_greedy, decays[0])
    pi_greedy = RandomPolicy()
    rez = np.zeros((C["create_data"]["N_trajs"], 4))
    rm = ReplayMemory(100000, TransitionGym)
    for i in range(C["create_data"]["N_trajs"]):
        if i % 50 == 0:
            logger.info(i)
        pi_epsilon_greedy.epsilon = decays[i]
        pi_epsilon_greedy.pi_greedy = pi_greedy
        trajectory, rew_r, rew_c, ret_r, ret_c = urpy.execute_policy_one_dialogue(
            e, pi_epsilon_greedy, gamma_r=C["gamma"], gamma_c=C["gamma_c"], beta=1.0, print_dial=False)
        rez[i] = np.array([rew_r, rew_c, ret_r, ret_c])
        for sample in trajectory:
            rm.push(*sample)
        if i > 0 and i % C["create_data"]["trajs_by_ftq_batch"] == 0:
            logger.info("------------------------------------------------------------------")
            logger.info("------------------------------------------------------------------")

            transitions_ftq, transition_bftq = urpy.datas_to_transitions(rm.memory, e, feature,
                                                                         C["create_data"]["lambda_"],
                                                                         C["create_data"]["normalize_reward"])
            logger.info("[LEARNING FTQ PI GREEDY] #samples={}".format(len(transitions_ftq)))

            ftq.reset(C["create_data"]["reset_weight"])
            pi = ftq.fit(transitions_ftq)
            pi_greedy = PytorchFittedPolicy(pi, e.action_space, e, feature)
            logger.info("------------------------------------------------------------------")
            logger.info("------------------------------------------------------------------")
    rm.save_memory(C.workspace + "/" + C.id + ".data")
    np.savetxt(C.workspace + "/" + C.id + ".results", rez)
    urpy.print_results(rez)


if __name__ == "__main__":
    C.load("config_main_pydial/test.json")
    main()
