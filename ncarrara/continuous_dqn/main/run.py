from ncarrara.continuous_dqn.main import generate_sources, learn_autoencoders, test_and_base, transfer_dqn
from ncarrara.continuous_dqn.tools.configuration import C
import sys


def main(config):
    C.load_pytorch().load(config)  # .create_fresh_workspace(force=True)

    generate_sources.main(
        source_envs=C["source_envs"],
        feature_dqn_info=C["feature_dqn_info"],
        net_params=C["net_params"],
        dqn_params=C["dqn_params"],
        seed=C.seed,
        device=C.device,
        workspace=C.path_sources,
        **C["generate_sources"]
    )
    learn_autoencoders.main(
        feature_autoencoder_info=C["feature_autoencoder_info"],
        workspace=C.path_sources,
        device=C.device,
        **C["learn_autoencoders"])

    test_and_base.main(loss_function_str=C["learn_autoencoders"]["loss_function_str"],
                       feature_autoencoder_info=C["feature_autoencoder_info"],
                       target_envs=C["target_envs"],
                       N=C["test_and_base"]["N"],
                       path_models=C.path_sources / "ae",
                       path_samples=C.path_sources / "samples",
                       seed=C.seed,
                       source_params=C.load_sources_params(),
                       device=C.device)
    # transfer_dqn.main()
    # transfer_dqn.show()


if __name__ == "__main__":
    config = sys.argv[1]
    print(config)
    main(config)
