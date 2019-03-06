from ncarrara.continuous_dqn.dqn.utils_dqn import run_dqn
from ncarrara.continuous_dqn.main import plot_data
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
        feature_dqn_info, N, source_params, device, loss_function_autoencoder_str, traj_max_size, configs):
    epsilon_decay(start_decay, decay, N, savepath=workspace)
    envs, tests_params = generate_envs(**target_envs)
    feature_autoencoder = build_feature_autoencoder(feature_autoencoder_info)
    feature_dqn = build_feature_dqn(feature_dqn_info)
    autoencoders = utils.load_autoencoders(workspace / "sources" / "ae", device)
    # ers = utils.load_memories(workspace, C["generate_samples"]["as_json"])

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

    # results_w_t = np.zeros((len(envs), N))
    # results_w_t_greedy = np.zeros((len(envs), N))
    datas = []
    import collections
    configs = collections.OrderedDict(configs)
    # data = pd.DataFrame()
    for i_env in range(len(envs)):
        logger.info("======================= ENV TARGET {} ======================".format(i_env))
        test_params = tests_params[i_env]
        logger.info(test_params)
        # env_data = pd.DataFrame()

        for key, config in configs.items():
            logger.info("=== config {} ====".format(key))
            if config["selection_method"] != "no_transfer":
                transfer_params_config = transfer_params
                transfer_params_config["test_params"] = test_params
                transfer_params_config["selection_method"] = config["selection_method"]
                transfer_params_config["transfer_network_type"] = config["transfer_network_type"]
            else:
                transfer_params_config = None
            env = envs[i_env]
            ret, ret_greedy, _ = run_dqn(
                env,
                workspace=workspace / key,
                seed=seed,
                device=device,
                feature_dqn=feature_dqn,
                net_params=net_params,
                dqn_params=dqn_params,
                N=N,
                decay=decay,
                start_decay=start_decay,
                transfer_params=transfer_params_config,
                traj_max_size=traj_max_size
            )
            # config_data = pd.DataFrame({"greedy": ret_greedy, "not_greedy": ret})
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>><")
            # print(config_data)
            # env_data = env_data.append(config_data)
            datas.append(ret_greedy)
            datas.append(ret)

        # print("xoxoxox")
        # print(env_data)
        # data = data.append(env_data)
    # print(data)
    midx = pd.MultiIndex.from_product([
        [i_env for i_env in range(len(envs))],
        [key for key, config in configs.items()],
        [True, False]], names=["env", "config", "is_greedy"])
    print(midx)
    xaxa = pd.DataFrame(datas, index=midx)  # , columns=["env", "config", "greedy"])
    print(xaxa)
    xaxa.to_pickle(workspace / "data.pd")
    # plot_data.main(
    #     np.asarray(r_w_t),
    #     np.asarray(r_w_t_greedy),
    #     workspace=workspace,
    #     show_all=False)
    # np.savetxt(workspace / "results", results_w_t)
    # np.savetxt(workspace / "results_greedy", results_w_t_greedy)
    # with open(workspace / "target" / "params.json", 'w') as file:
    #     dump = json.dumps(tests_params, indent=4)
    #     print(dump)
    #     file.write(dump)
