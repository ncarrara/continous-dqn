# coding=utf-8
from ncarrara.bftq_pydial.tools.configuration import C
from ncarrara.bftq_pydial.tools.features import feature_factory
from ncarrara.utils.math import epsilon_decay, set_seed
from ncarrara.utils.os import makedirs
from ncarrara.utils_rl.algorithms.pytorch_fittedq import NetFTQ, PytorchFittedQ
from ncarrara.utils_rl.environments.envs_factory import generate_envs
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.bftq_pydial.tools.policies import PytorchFittedPolicy, RandomPolicy
import ncarrara.bftq_pydial.tools.utils_run_pydial as urpy
from ncarrara.bftq_pydial.tools.policies import EpsilonGreedyPolicy
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(config_file, lambda_):
    C.load(config_file).load_pytorch().create_fresh_workspace(force=True)
    envs, params = generate_envs(**C["generate_envs"])
    e = envs[0]
    set_seed(C.seed, e)

    feature = feature_factory(C["feature_str"])

    policy_network = NetFTQ(n_in=len(feature(e.reset(), e)), n_out=e.action_space.n, **C["net_params"])

    ftq = PytorchFittedQ(
        device=C.device,
        action_str=None if not hasattr("action_str", e) else e.action_str,
        policy_network=policy_network,
        test_policy=None,
        gamma=C["gamma"],
        **C["ftq_params"]
    )

    pi_greedy = None

    decays = epsilon_decay(**C["learn_ftq_egreedy"]["epsilon_decay"],
                           N=C["learn_ftq_egreedy"]["N_trajs"],
                           savepath=C.path_learn_ftq_egreedy)

    pi_random = RandomPolicy()
    pi_epsilon_greedy = EpsilonGreedyPolicy(pi_greedy, decays[0], pi_random=pi_random)
    pi_greedy = pi_random
    rez = np.zeros((C["learn_ftq_egreedy"]["N_trajs"], 4))
    rm = Memory()
    batch = 0

    for i_traj in range(C["learn_ftq_egreedy"]["N_trajs"]):
        if i_traj % 50 == 0: logger.info(i_traj)
        pi_epsilon_greedy.epsilon = decays[i_traj]
        pi_epsilon_greedy.pi_greedy = pi_greedy
        trajectory, rew_r, rew_c, ret_r, ret_c = urpy.execute_policy_one_dialogue(
            e, pi_epsilon_greedy, gamma_r=C["gamma"], gamma_c=C["gamma_c"])
        rez[i_traj] = np.array([rew_r, rew_c, ret_r, ret_c])
        for sample in trajectory:
            rm.push(*sample)
        if i_traj > 0 and (i_traj + 1) % C["learn_ftq_egreedy"]["trajs_by_ftq_batch"] == 0:
            transitions_ftq, _ = urpy.datas_to_transitions(
                rm.memory, e, feature, lambda_, C["learn_ftq_egreedy"]["normalize_reward"])
            logger.info("[BATCH={}]---------------------------------------".format(batch))
            logger.info("[BATCH={}][learning ftq pi greedy] #samples={} #traj={}"
                        .format(batch, len(transitions_ftq), i_traj + 1))
            logger.info("[BATCH={}]---------------------------------------".format(batch))
            ftq.reset(True)
            ftq.workspace = C.path_learn_ftq_egreedy + "/batch={}".format(batch)
            makedirs(ftq.workspace)
            pi = ftq.fit(transitions_ftq)
            ftq.save_policy()
            os.system("cp {}/policy.pt {}/final_policy.pt".format(ftq.workspace, C.path_learn_ftq_egreedy))
            pi_greedy = PytorchFittedPolicy(pi, e, feature)
            batch += 1

    ftq.workspace = C.path_learn_ftq_egreedy


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        config_file = sys.argv[1]
        lambda_ = float(sys.argv[2])
    else:
        config_file = "config/camera_ready_999.json"
        lambda_ = 0.
    main(config_file, lambda_)
