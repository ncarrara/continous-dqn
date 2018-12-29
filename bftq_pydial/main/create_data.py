# coding=utf-8
from bftq_pydial.tools.features import feature_0
from utils.math import epsilon_decay
from utils_rl.transition.replay_memory import ReplayMemory
from bftq_pydial.tools.policies import PytorchFittedPolicy, RandomPolicy
import utils_rl.algorithms.pytorch_fittedq as pftq
import numpy as np
from gym_pydial.env.env_pydial import EnvPydial
import bftq_pydial.tools.utils_run_pydial as urpy
from bftq_pydial.tools.policies import EpsilonGreedyPolicy
from bftq_pydial.tools.configuration import C
from utils_rl.transition.transition import TransitionGym


def main():
    e = EnvPydial(config_file=C.pydial_configuration, error_rate=0.3)
    print(e.cl)
    for k, v in e.cl.items():
        v.setLevel("ERROR")

    e.reset()
    action_space = e.action_space()
    feature = feature_0

    def process_between_epoch(pi):
        print("process_between_epoch ...")
        pi = PytorchFittedPolicy(pi, action_space, e, feature)
        _, results = urpy.execute_policy(e, pi, C["create_data"]["ftq_params"]["gamma"], C["create_data"]["gamma_c"], 5,
                                         1., False)
        return np.mean(results, axis=0)

    size_state = len(feature(e.reset(), e))
    print("neural net input size :", size_state)

    policy_network = pftq.NetFTQ(n_in=size_state, n_out=len(action_space), **C["create_data"]["net_params"])

    ftq = pftq.PytorchFittedQ(
        action_str=e.action_space_str(),
        policy_network=policy_network,
        process_between_epoch=process_between_epoch, **C["create_data"]["ftq_params"]
    )

    pi_greedy = None

    decays = epsilon_decay(**C["create_data"]["epsilon_decay"], N=C["create_data"]["N_trajs"])

    pi_epsilon_greedy = EpsilonGreedyPolicy(pi_greedy, decays[0])
    pi_greedy = RandomPolicy()
    rez = np.zeros((C["create_data"]["N_trajs"], 4))
    rm = ReplayMemory(100000, TransitionGym)
    for i in range(C["create_data"]["N_trajs"]):
        if i % 10 == 0:
            print(i)
        pi_epsilon_greedy.epsilon = decays[i]
        pi_epsilon_greedy.pi_greedy = pi_greedy
        trajectory, rew_r, rew_c, ret_r, ret_c = urpy.execute_policy_one_dialogue(
            e, pi_epsilon_greedy, gamma_r=C["main"]["ftq_params"]["gamma"],
            gamma_c=C["create_data"]["gamma_c"],
            beta=1.0, print_dial=False)
        rez[i] = np.array([rew_r, rew_c, ret_r, ret_c])
        for sample in trajectory:
            rm.push(*sample)
        if i > 0 and i % C["create_data"]["trajs_by_ftq_batch"] == 0:
            transitions_ftq, transition_bftq = urpy.datas_to_transitions(rm.memory, e, feature,
                                                                         C["create_data"]["lambda_"],
                                                                         C["create_data"]["normalize_reward"])
            print("[LEARNING FTQ PI GREEDY] #samples={}".format(len(transitions_ftq)))
            ftq.reset(C["create_data"]["reset_weight"])
            pi = ftq.fit(transitions_ftq)
            pi_greedy = PytorchFittedPolicy(pi, action_space, e, feature)
    rm.save_memory(C.workspace + "/" + C.id + ".data")
    urpy.print_results(rez)


if __name__ == "__main__":
    C.load("config_main_pydial/test.json")
    main()
