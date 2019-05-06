from pathlib import Path

from ncarrara.continuous_dqn.dqn.action_transfer_module import ActionTransferModule
from ncarrara.continuous_dqn.dqn.bellman_transfer_module import BellmanTransferModule
from ncarrara.continuous_dqn.dqn.utils_dqn import run_dqn
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
        start_decay, decay, feature_dqn_info, N,
        source_params, device,
        traj_max_size, configs, gamma,
        writer=None, path_sources=None, **kwargs):
    epsilon_decay(start_decay, decay, N, savepath=workspace)

    envs, tests_params = generate_envs(**target_envs)

    feature_dqn = build_feature_dqn(feature_dqn_info)

    transfer_params = {
        "device": device,
        "evaluate_continuously": False,
        "sources_params": source_params,
    }

    datas = []
    import collections
    configs = collections.OrderedDict(configs)
    # data = pd.DataFrame()
    for i_env in range(len(envs)):
        logger.info("===================    ==== ENV TARGET {} ======================".format(i_env))
        test_params = tests_params[i_env]
        logger.info(test_params)
        # env_data = pd.DataFrame()

        for key, config in configs.items():
            logger.info("=== config {} ====".format(key))
            logger.info("config : {}".format(config))
            if config["selection_method"] != "no_transfer":
                transfer_params_config = {**transfer_params, **config}
                transfer_params_config["test_params"] = test_params
                if config["type_transfer_module"] == "BellmanTransferModule":
                    transfer_params_config["Q_sources"] = utils.load_q_sources(Path(path_sources) / "dqn", device)
                    transfer_module = BellmanTransferModule(**transfer_params_config)

                elif config["type_transfer_module"] == "AutoencoderTransferModule":
                    # feature_autoencoder = build_feature_autoencoder(feature_autoencoder_info)
                    # autoencoders = utils.load_autoencoders(Path(path_sources) / "ae", device)
                    # "loss_autoencoders": tu.loss_fonction_factory(loss_function_autoencoder_str),
                    raise NotImplementedError()
                elif config["type_transfer_module"] == "ActionTransferModule":
                    transfer_params_config["Q_sources"] = utils.load_q_sources(Path(path_sources) / "dqn", device)
                    transfer_params_config["Q_base_target"] = None
                    transfer_module = ActionTransferModule(**transfer_params_config)
                else:
                    raise Exception(
                        "Unknow type_transfer_module={}".format(transfer_params_config["type_transfer_module"]))
            else:
                transfer_module = None
            env = envs[i_env]
            ret, ret_greedy, _, _ = run_dqn(
                env,
                id="{}_e{}".format(key,i_env),
                workspace=workspace / key,
                seed=seed,
                device=device,
                feature_dqn=feature_dqn,
                net_params=net_params,
                dqn_params=dqn_params,
                N=N,
                decay=decay,
                start_decay=start_decay,
                transfer_module=transfer_module,
                traj_max_size=traj_max_size,
                writer=writer,
                gamma=gamma
            )
            datas.append(ret_greedy)
            datas.append(ret)

    midx = pd.MultiIndex.from_product([
        [i_env for i_env in range(len(envs))],
        [key for key, config in configs.items()],
        [True, False]], names=["env", "config", "is_greedy"])
    xaxa = pd.DataFrame(datas, index=midx)
    xaxa.to_pickle(workspace / "data.pd")
