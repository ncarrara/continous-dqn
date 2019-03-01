from ncarrara.continuous_dqn.main import generate_samples, learn_autoencoders, test_and_base, transfer_dqn
from ncarrara.continuous_dqn.tools.configuration import C
import sys


def main(config):
    # C.load_pytorch().load("config/tests/main.json")#.create_fresh_workspace(force=True)
    C.load_pytorch().load(config)  # .create_fresh_workspace(force=True)
    # C.load_pytorch().load("config/0_slot_filling.json")
    # C.load("config/0_pydial.json").load_pytorch()#.create_fresh_workspace()

    # generate_samples.main()
    # learn_autoencoders.main(**C["learn_autoencoders"])
    # test_and_base.main(loss_function_str=C["learn_autoencoders"]["loss_function_str"],
    #                    feature_autoencoder_info=C["feature_autoencoder_info"],
    #                    target_envs=C["target_envs"],
    #                    N=C["test_and_base"]["N"],
    #                    path_models=C.path_models,
    #                    path_samples=C.path_samples,
    #                    seed=C.seed,
    #                    source_params=C.load_sources_params(),
    #                    device=C.device)
    transfer_dqn.main()
    # transfer_dqn.show()


if __name__ == "__main__":
    config = sys.argv[1]
    print(config)
    main(config)
