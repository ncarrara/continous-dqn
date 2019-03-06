from ncarrara.budgeted_rl.main.utils import abstract_main
from ncarrara.continuous_dqn.main import generate_sources, learn_autoencoders, test_and_base, transfer_dqn, plot_data
import sys


def main(config):
    if "generate_sources" in config:
        generate_sources.main(
            source_envs=config["source_envs"],
            feature_dqn_info=config["feature_dqn_info"],
            net_params=config["net_params"],
            dqn_params=config["dqn_params"],
            seed=config.seed,
            device=config.device,
            workspace=config.path_sources,
            **config["generate_sources"]
        )
        learn_autoencoders.main(
            feature_autoencoder_info=config["feature_autoencoder_info"],
            workspace=config.path_sources,
            device=config.device,
            **config["learn_autoencoders"])

        test_and_base.main(
            loss_autoencoders_str=config["learn_autoencoders"]["loss_function_str"],
            feature_autoencoder_info=config["feature_autoencoder_info"],
            target_envs=config["target_envs"],
            N=config["test_and_base"]["N"],
            path_models=config.path_sources / "ae",
            path_samples=config.path_sources / "samples",
            seed=config.seed,
            source_params=config.load_sources_params(),
            device=config.device)

    transfer_dqn.main(
        workspace=config.workspace,
        seed=config.seed,
        target_envs=config["target_envs"],
        net_params=config["net_params"],
        dqn_params=config["dqn_params"],
        source_params=config.load_sources_params(),
        device=config.device,
        feature_autoencoder_info=config["feature_autoencoder_info"],
        feature_dqn_info=config["feature_dqn_info"],
        loss_function_autoencoder_str=config["learn_autoencoders"]["loss_function_str"],
        **config["transfer_dqn"]
    )
    plot_data.main(workspace=config.workspace, show_all=True)


if __name__ == "__main__":
    from ncarrara.continuous_dqn.tools.configuration import C
    seeds = None
    override_device_str = None
    print(sys.argv)
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if len(sys.argv) > 2:
            seed_start = int(sys.argv[2])
            seed_end = int(sys.argv[3])
            seeds = range(seed_start, seed_end)
            if len(sys.argv) > 4:
                override_device_str = sys.argv[4]
    else:
        config_file = "../config/test_egreedy.json"
        C.load(config_file).create_fresh_workspace(force=True)
        seeds = [0, 1]

    override_param_grid = {}
    if seeds is not None:
        override_param_grid['general.seed'] = seeds

    abstract_main.main(C,config_file, override_param_grid, override_device_str, main)
