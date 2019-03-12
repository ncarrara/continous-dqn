from pathlib import Path

from ncarrara.continuous_dqn.dqn.utils_dqn import run_dqn
from ncarrara.continuous_dqn.main import plot_data, plot_all_data
from ncarrara.utils.os import makedirs
from ncarrara.utils_rl.environments.envs_factory import generate_envs
from ncarrara.continuous_dqn.tools import utils
import pandas as pd
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
        feature_dqn_info, N, source_params, device, loss_function_autoencoder_str, traj_max_size, configs, gamma,
        writer=None, path_sources=None):
    epsilon_decay(start_decay, decay, N, savepath=workspace)

    envs, tests_params = generate_envs(**target_envs)
    feature_autoencoder = build_feature_autoencoder(feature_autoencoder_info)
    feature_dqn = build_feature_dqn(feature_dqn_info)
    if path_sources is None:
        autoencoders = utils.load_autoencoders(workspace / "sources" / "ae", device)
        # ers = utils.load_memories(workspace, C["generate_samples"]["as_json"])
        Q_sources = utils.load_q_sources(workspace / "sources" / "models_dqn", device)
    else:
        autoencoders = utils.load_autoencoders(Path(path_sources) / "ae", device)
        # ers = utils.load_memories(workspace, C["generate_samples"]["as_json"])
        Q_sources = utils.load_q_sources(Path(path_sources) / "models_dqn", device)
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

    datas = []
    import collections
    configs = collections.OrderedDict(configs)
    for i_env in range(len(envs)):
        logger.info("===================    ==== ENV TARGET {} ======================".format(i_env))
        test_params = tests_params[i_env]
        logger.info(test_params)
        name_folder = "target_env_{}".format(i_env)
        ws_env = workspace / name_folder
        for key, config in configs.items():

            ws = ws_env / key
            makedirs(ws)
            logger.info("=== config {} ====".format(key))
            if config["selection_method"] != "no_transfer":
                transfer_params_config = transfer_params
                transfer_params_config["test_params"] = test_params
                transfer_params_config["selection_method"] = config["selection_method"]
            else:
                transfer_params_config = None
            env = envs[i_env]
            ret, ret_greedy, _, dqn = run_dqn(
                env,
                workspace=ws,
                seed=seed,
                device=device,
                feature_dqn=feature_dqn,
                net_params=net_params,
                dqn_params=dqn_params,
                N=N,
                decay=decay,
                start_decay=start_decay,
                transfer_params=transfer_params_config,
                traj_max_size=traj_max_size,
                writer=writer,
                gamma=gamma
            )
            with open(ws / "stats.bin", 'w') as f:
                import json
                json.dump(dqn.stats, f)
            for key_stats, values in dqn.stats.items():
                import matplotlib.pyplot as plt
                if values:
                    plt.plot(range(len(values)), values)
                    plt.title(key_stats)
                    plt.savefig(ws / key_stats)
                    plt.close()
            datas.append(ret_greedy)
            datas.append(ret)

        # partial saving
        midx = pd.MultiIndex.from_product([
            [i_e for i_e in range(i_env + 1)],
            [key for key, config in configs.items()],
            [True, False]], names=["env", "config", "is_greedy"])
        xaxa = pd.DataFrame(datas, index=midx)
        xaxa.to_pickle(workspace / "data.pd")
        plot_data.main(workspace)
