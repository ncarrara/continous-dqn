from ncarrara.continuous_dqn.dqn.transfer_module import TransferModule
from ncarrara.continuous_dqn.tools.utils import load_memories
from ncarrara.utils.os import makedirs
from ncarrara.utils.torch_utils import loss_fonction_factory
from ncarrara.utils_rl.environments.envs_factory import generate_envs
from ncarrara.continuous_dqn.tools import utils
import logging
import numpy as np

from ncarrara.continuous_dqn.tools.features import build_feature_autoencoder

logger = logging.getLogger(__name__)


def main(loss_autoencoders_str, feature_autoencoder_info, target_envs, N, path_models, path_samples, seed,
         source_params, device, workspace):
    makedirs(workspace)
    loss_autoencoders = loss_fonction_factory(loss_autoencoders_str)
    autoencoders = utils.load_autoencoders(path_models, device)
    feature_autoencoder = build_feature_autoencoder(feature_autoencoder_info)
    tm = TransferModule(
        autoencoders=autoencoders,
        loss_autoencoders=loss_autoencoders,
        feature_autoencoders=feature_autoencoder,
        device=device,
        sources_params=source_params)
    errors_base = []
    # all_transitions = utils.read_samples_for_autoencoders(path_samples, feature_autoencoder)
    memories = load_memories(path_samples, as_json=False)
    for memory in memories:
        tm.reset()
        tm.push_memory(memory.memory)
        tm.evaluate()
        errors_base.append(tm.errors)

    print("================================================ base ================================================")
    print(errors_base)
    print(utils.array_to_cross_comparaison(errors_base, source_params, source_params))

    test_envs, test_params = generate_envs(**target_envs)

    errors_test = []
    for ienv, test_env in enumerate(test_envs):
        tm.reset()
        if seed is not None:
            test_env.seed(seed)
        for _ in range(N):
            s = test_env.reset()
            done = False
            while not done:
                if hasattr(test_env, "action_space_executable"):
                    a = np.random.choice(test_env.action_space_executable())
                else:
                    a = test_env.action_space.sample()
                s_, r_, done, info = test_env.step(a)
                tm.push(s, a, r_, s_, done, info)
                s = s_
        tm.evaluate()
        errors_test.append(tm.errors)

    print("================================================ test ================================================")
    print(utils.array_to_cross_comparaison(errors_test, source_params, test_params))

    import pickle

    with open(workspace / "errors_base.bin", 'wb') as f:
        pickle.dump(errors_base, f)

    with open(workspace / "errors_test.bin", 'wb') as f:
        pickle.dump(errors_test, f)

    with open(workspace / "source_params.bin", 'wb') as f:
        pickle.dump(source_params, f)

    with open(workspace / "test_params.bin", 'wb') as f:
        pickle.dump(test_params, f)

# if __name__ == "__main__":
#     # execute only if run as a script
#     C.load("config/0_pydial.json").load_pytorch()
#     main()
