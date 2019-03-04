from ncarrara.continuous_dqn.dqn.utils_dqn import run_dqn
from ncarrara.continuous_dqn.main import plot_data
from ncarrara.utils_rl.environments.envs_factory import generate_envs
from ncarrara.continuous_dqn.tools import utils

import numpy as np
import logging

from ncarrara.continuous_dqn.tools.features import build_feature_autoencoder, build_feature_dqn
from ncarrara.utils.math_utils import epsilon_decay

logger = logging.getLogger(__name__)
import json
import ncarrara.utils.torch_utils as tu


def main(
        workspace, seed, target_envs, net_params, dqn_params,
        start_decay, decay, feature_autoencoder_info,
        feature_dqn_info, N, source_params, device, loss_function_autoencoder_str,traj_max_size):
    epsilon_decay(start_decay, decay, N, savepath=workspace)
    envs, tests_params = generate_envs(**target_envs)
    feature_autoencoder = build_feature_autoencoder(feature_autoencoder_info)
    feature_dqn = build_feature_dqn(feature_dqn_info)
    autoencoders = utils.load_autoencoders(workspace / "sources" / "ae", device)
    # ers = utils.load_memories(C.path_samples, C["generate_samples"]["as_json"])

    Q_sources = utils.load_q_sources(workspace / "sources" / "dqn", device)

    # for er in ers:
    #     er.apply_feature_to_states(feature_dqn)
    #     er.to_tensors(C.device)

    transfer_params = {
        "autoencoders": autoencoders,
        "loss_autoencoders": tu.loss_fonction_factory(loss_function_autoencoder_str),

        "experience_replays": None,

        "feature_autoencoders": feature_autoencoder,
        "Q_sources": Q_sources,
        "device": device,
        "evaluate_continuously": False,
        "sources_params": source_params,

    }

    results_w_t = np.zeros((len(envs), N))
    results_wo_t = np.zeros((len(envs), N))
    results_w_t_greedy = np.zeros((len(envs), N))
    results_wo_t_greedy = np.zeros((len(envs), N))
    for i_env in range(len(envs)):
        test_params = tests_params[i_env]
        transfer_params["test_params"] = test_params
        env = envs[i_env]
        logger.info("============================================================")
        logger.info("======================= ENV TARGET {} ======================".format(i_env))
        logger.info("============================================================")
        logger.info(test_params)

        logger.info("======== WITH TRANSFER ==========")

        r_w_t, r_w_t_greedy, _ = run_dqn(
            env,
            workspace=workspace / "target" / "w_transfer",
            seed=seed,
            device=device,
            feature_dqn=feature_dqn,
            net_params=net_params,
            dqn_params=dqn_params,
            N=N,
            decay=decay,
            start_decay=start_decay,
            transfer_params=transfer_params,
            traj_max_size=traj_max_size
        )

        logger.info("======== WITHOUT TRANSFER ==========")
        r_wo_t, r_wo_t_greedy, _ = run_dqn(
            env,
            workspace=workspace / "target" / "wo_transfer",
            seed=seed,
            device=device,
            feature_dqn=feature_dqn,
            net_params=net_params,
            dqn_params=dqn_params,
            N=N,
            decay=decay,
            start_decay=start_decay,
            traj_max_size=traj_max_size
        )

        results_w_t[i_env] = r_w_t
        results_wo_t[i_env] = r_wo_t
        results_w_t_greedy[i_env] = r_w_t_greedy
        results_wo_t_greedy[i_env] = r_wo_t_greedy
        plot_data.main(
            np.asarray(r_w_t),
            np.asarray(r_wo_t),
            np.asarray(r_w_t_greedy),
            np.asarray(r_wo_t_greedy),
            workspace=workspace,
            show_all=False)
        # exit()
    np.savetxt(workspace / "target" / "w_transfer" / "results", results_w_t)
    np.savetxt(workspace / "target" / "wo_transfer" / "results", results_wo_t)
    np.savetxt(workspace / "target" / "w_transfer" / "results_greedy", results_w_t_greedy)
    np.savetxt(workspace / "target" / "wo_transfer" / "results_greedy", results_wo_t_greedy)
    with open(workspace / "target" / "params.json", 'w') as file:
        dump = json.dumps(tests_params, indent=4)
        print(dump)
        file.write(dump)
