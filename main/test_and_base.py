from dqn.transfer_module import TransferModule
from configuration import C
import utils
from envs.envs_factory import generate_envs
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def main():
    N = 1000
    autoencoders = utils.load_autoencoders(C.workspace + "/" + C.path_models)
    all_transitions = utils.read_samples(C.workspace + "/" + C.path_samples)

    tm = TransferModule(models=autoencoders, loss=F.l1_loss)

    envs, params = generate_envs(**C["generate_samples"])
    seed = C["general"]["seed"]

    errors_test = []
    errors_base = []
    for ienv, env in enumerate(envs):
        tm.reset()
        logger.info("evaluating base set for env {}".format(ienv))
        errors_base.append(tm._errors_with_tensors(all_transitions[ienv]))
        logger.info("evaluating test set for env {}".format(ienv))
        utils.set_seed(seed=seed, env=env)
        env.seed(seed)
        for _ in range(N):
            s = env.reset()
            done = False
            while not done:
                a = env.action_space.sample()
                s_, r_, done, info = env.step(a)
                tm.push(s.tolist(), a, r_, s_.tolist(),done,info)

                s = s_
        errors_test.append(tm.errors())
    print("base")
    print(utils.array_to_cross_comparaison(errors_base))
    print("test")
    print(utils.array_to_cross_comparaison(errors_test))


if __name__ == "__main__":
    # execute only if run as a script
    C.load("config/0.json")
    main()
