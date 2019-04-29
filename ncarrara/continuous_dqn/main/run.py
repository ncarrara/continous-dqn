from pathlib import Path
from pprint import pprint

from ncarrara.budgeted_rl.main.utils import abstract_main
from ncarrara.continuous_dqn.main import generate_sources, learn_autoencoders, test_and_base, transfer_dqn, plot_data
import sys

from ncarrara.utils.os import empty_directory, makedirs


def main(config):
    if "generate_sources" in config:
        generate_sources.main(
            source_envs=config["source_envs"],
            feature_dqn_info=config["feature_dqn_info"],
            net_params=config["net_params"],
            dqn_params=config["dqn_params"],
            gamma=config["dqn_params"]["gamma"],
            seed=config.seed,
            device=config.device,
            workspace=config.path_sources,
            writer=config.writer,
            **config["generate_sources"]
        )
    if "learn_autoencoders" in config:
        learn_autoencoders.main(
            feature_autoencoder_info=config["feature_autoencoder_info"],
            workspace=config.path_sources,
            device=config.device,
            N_actions=config["N_actions"],
            type_ae = config["feature_autoencoder_info"]["type_ae"],
            writer=config.writer,
            **config["learn_autoencoders"])
    if "test_and_base" in config:
        test_and_base.main(
            loss_autoencoders_str=config["learn_autoencoders"]["loss_function_str"],
            feature_autoencoder_info=config["feature_autoencoder_info"],
            target_envs=config["target_envs"],
            N=config["test_and_base"]["N"],
            path_models=config.path_sources / "ae",
            path_samples=config.path_sources / "samples",
            seed=config.seed,
            N_actions=config["N_actions"],
            source_params=config.load_sources_params(),
            device=config.device)
    if "transfer_dqn" in config:
        # if config["transfer_dqn"]["path_sources"] is not None:
        #     config.path_sources =  Path(config["transfer_dqn"]["path_sources"])
        transfer_dqn.main(
            workspace=config.workspace,
            seed=config.seed,
            target_envs=config["target_envs"],
            net_params=config["net_params"],
            dqn_params=config["dqn_params"],
            gamma=config["dqn_params"]["gamma"],
            source_params=config.load_sources_params(),
            device=config.device,
            path_sources=config.path_sources,
            feature_dqn_info=config["feature_dqn_info"],
            writer=config.writer,
            **config["transfer_dqn"]
        )
        plot_data.main(workspace=config.workspace)


if __name__ == "__main__":
    from ncarrara.continuous_dqn.tools.configuration import C

    seeds = None
    override_device_str = None
    print(sys.argv)
    if len(sys.argv) > 2:
        config_file = sys.argv[1]
        seed_start = int(sys.argv[2])
        seed_end = int(sys.argv[3])
        seeds = range(seed_start, seed_end)
        override_param_grid = {}
        if seeds is not None:
            override_param_grid['general.seed'] = seeds

        abstract_main.main(C, config_file, override_param_grid, override_device_str, main)
    # else:
    #     # DEBUGGING
    #     import os
    #
    #     seed = 0
    #     config_file = sys.argv[1]
    #     config_file_sources = "config/cartpole/easy2.json"
    #
    #     with open(config_file, 'r') as infile:
    #         import json
    #
    #         dict_debug = json.load(infile)
    #
    #
    #     makedirs(dict_debug["general"]["workspace"])
    #
    #     with open(config_file_sources, 'r') as infile:
    #         import json
    #
    #         dict = json.load(infile)
    #         os.system("cp -r {}/{} {}".format(dict["general"]["workspace"], seed, dict_debug["general"]["workspace"]))
    #
    #         dict_debug["general"]["workspace"] = dict_debug["general"]["workspace"] + "/" + str(seed)
    #         dict_debug["general"]["seed"] = seed
    #
    #     if "matplotlib_backend" in dict_debug["general"]:
    #         backend = dict_debug["general"]["matplotlib_backend"]
    #     else:
    #         backend = "Agg"
    #     C.load(dict_debug).load_pytorch(override_device_str).load_matplotlib(backend)
    #     pprint(dict_debug)
    #     main(C)
