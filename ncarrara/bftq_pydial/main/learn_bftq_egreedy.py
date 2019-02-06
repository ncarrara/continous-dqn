# coding=utf-8
from ncarrara.bftq_pydial.bftq.pytorch_budgeted_fittedq import PytorchBudgetedFittedQ, NetBFTQ
from ncarrara.bftq_pydial.tools.configuration import C
# C.load_matplotlib('agg')
from ncarrara.bftq_pydial.tools.features import feature_factory
from ncarrara.utils.math import epsilon_decay, set_seed
from ncarrara.utils.os import makedirs
from ncarrara.utils_rl.algorithms.pytorch_fittedq import NetFTQ, PytorchFittedQ
from ncarrara.utils_rl.environments.envs_factory import generate_envs
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.bftq_pydial.tools.policies import PytorchFittedPolicy, RandomPolicy, HandcraftedSlotFillingEnv, \
    RandomBudgetedPolicy, PytorchBudgetedFittedPolicy
import ncarrara.bftq_pydial.tools.utils_run_pydial as urpy
from ncarrara.bftq_pydial.tools.policies import EpsilonGreedyPolicy

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    envs, params = generate_envs(**C["generate_envs"])
    e = envs[0]
    set_seed(C.seed, e)

    feature = feature_factory(C["feature_str"])

    size_state = len(feature(e.reset(), e))

    logger.info("neural net input size : {}".format(size_state))

    import numpy as np
    betas = eval(C["betas"])
    if "betas_for_discretisation" in C.dict:
        betas_for_discretisation = eval(C["betas_for_discretisation"])
    else:
        betas_for_discretisation = betas

    action_str = getattr(e, "action_space_str", list(map(str, range(e.action_space.n))))

    policy_network_bftq = NetBFTQ(size_state=size_state,
                                  layers=C["bftq_net_params"]["intra_layers"] + [2 * e.action_space.n],
                                  **C["bftq_net_params"])

    bftq = PytorchBudgetedFittedQ(
        device=C.device,
        workspace=C.path_learn_bftq_egreedy + "/batch=0",
        betas=betas,
        betas_for_discretisation=betas_for_discretisation,
        N_actions=e.action_space.n,
        actions_str=action_str,
        policy_network=policy_network_bftq,
        gamma=C["gamma"],
        gamma_c=C["gamma_c"],
        **C["bftq_params"],

    )

    pi_greedy = None

    decays = epsilon_decay(**C["learn_bftq_egreedy"]["epsilon_decay"],
                           N=C["learn_bftq_egreedy"]["N_trajs"],
                           savepath=C.path_learn_bftq_egreedy)
    logger.info("decays={}".format(decays))

    pi_random = RandomBudgetedPolicy()
    pi_epsilon_greedy = EpsilonGreedyPolicy(pi_greedy, decays[0], pi_random=pi_random)
    pi_greedy = pi_random
    rez = np.zeros((C["learn_bftq_egreedy"]["N_trajs"], 4))
    rm = Memory()
    batch = 0

    for i_traj in range(C["learn_bftq_egreedy"]["N_trajs"]):
        if i_traj % 50 == 0: logger.info(i_traj)
        pi_epsilon_greedy.epsilon = decays[i_traj]
        pi_epsilon_greedy.pi_greedy = pi_greedy
        beta = np.random.sample()
        trajectory, rew_r, rew_c, ret_r, ret_c = urpy.execute_policy_one_dialogue(
            e, pi_epsilon_greedy, gamma_r=C["gamma"], gamma_c=C["gamma_c"], beta=beta)
        rez[i_traj] = np.array([rew_r, rew_c, ret_r, ret_c])
        for sample in trajectory:
            rm.push(*sample)
        if i_traj > 0 and (i_traj+1) % C["learn_bftq_egreedy"]["trajs_by_ftq_batch"] == 0:
            transitions_ftq, transition_bftq = urpy.datas_to_transitions(rm.memory, e, feature, 0,
                                                                         C["learn_bftq_egreedy"]["normalize_reward"])
            logger.info("[BATCH={}]---------------------------------------".format(batch))
            logger.info("[BATCH={}][learning bftq pi greedy] #samples={} #traj={}".format(batch, len(transitions_ftq),i_traj+1))
            logger.info("[BATCH={}]---------------------------------------".format(batch))
            bftq.reset(True)
            bftq.workspace = C.path_learn_bftq_egreedy + "/batch={}".format(batch)
            makedirs(bftq.workspace)
            pi = bftq.fit(transition_bftq)
            bftq.save_policy()
            pi_greedy = PytorchBudgetedFittedPolicy(pi, e, feature)
            batch += 1

    # rm.save_memory(C.path_create_data_bftq, "/" + C["learn_bftq_egreedy"]["filename_data"],
    #                C["learn_bftq_egreedy"]["as_json"])
    bftq.workspace = C.path_learn_bftq_egreedy
    bftq.save_policy(policy_name="final_policy")
    # np.savetxt(C.workspace + "/" + C.id + ".results", rez)

    # _, rez = urpy.execute_policy(e, pi_greedy,
    #                              gamma_r=C["gamma"], gamma_c=C["gamma_c"],
    #                              beta=1.0, N_dialogues=100)
    # print("greedy results")
    # print(urpy.format_results(rez))


if __name__ == "__main__":
    C.load("config/test3.json").load_pytorch().create_fresh_workspace(force=True)
    main()
