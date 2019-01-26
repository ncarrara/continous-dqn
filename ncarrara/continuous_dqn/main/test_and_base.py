from ncarrara.continuous_dqn.dqn.transfer_module import TransferModule
from ncarrara.utils_rl.environments.envs_factory import generate_envs
from ncarrara.continuous_dqn.tools import utils
import torch.nn.functional as F
import logging
import numpy as np

from ncarrara.continuous_dqn.tools.configuration import C
from ncarrara.continuous_dqn.tools.features import build_feature_autoencoder

logger = logging.getLogger(__name__)


def main():
    autoencoders = utils.load_autoencoders(C.path_models)
    feature_autoencoder = build_feature_autoencoder(C["feature_autoencoder_info"])

    all_transitions = utils.read_samples_for_autoencoders(C.path_samples,feature_autoencoder)
    tm = TransferModule(models=autoencoders, loss=F.l1_loss)

    errors_base = []
    for transitions in all_transitions:
        tm.reset()
        errors_base.append(tm._errors_with_tensors(transitions))

    print("================================================ base ================================================")
    source_params = C.load_sources_params()
    print(utils.array_to_cross_comparaison(errors_base, source_params, source_params))


    test_envs, test_params = generate_envs(**C["target_envs"])
    # seed = C["general"]["seed"]
    errors_test = []
    for ienv, test_env in enumerate(test_envs):
        tm.reset()
        # utils.set_seed(seed=seed, env=test_env)
        if C.seed is not None:
            test_env.seed(C.seed)
        for _ in range(C["test_and_base"]["N"]):
            s = test_env.reset()
            done = False
            while not done:
                if hasattr(test_env, "action_space_executable"):
                    a = np.random.choice(test_env.action_space_executable())
                else:
                    a = test_env.action_space.sample()
                # a = test_env.action_space.sample()
                s_, r_, done, info = test_env.step(a)
                feat=feature_autoencoder((s, a, r_, s_,done,info))
                tm.push(feat)
                s = s_
        errors_test.append(tm.errors())

    print("================================================ test ================================================")
    print(utils.array_to_cross_comparaison(errors_test, source_params, test_params))


if __name__ == "__main__":
    # execute only if run as a script
    C.load("config/0_pydial.json").load_pytorch()
    main()
