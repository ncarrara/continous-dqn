# coding=utf-8
from ncarrara.budgeted_rl.tools.features import feature_factory
from ncarrara.utils.math_utils import set_seed
from ncarrara.utils.os import makedirs
from ncarrara.utils_rl.algorithms.pytorch_fittedq import NetFTQ, PytorchFittedQ
from ncarrara.budgeted_rl.tools.policies import PytorchBudgetedFittedPolicy, PytorchFittedPolicy
import ncarrara.budgeted_rl.tools.utils_run as urpy
from ncarrara.utils_rl.environments import envs_factory


def main(device, workspace, policy_path, generate_envs, ftq_net_params, ftq_params,
         feature_str, gamma, gamma_c, N_trajs, seed, lambda_,path_results, **args):
    envs, params = envs_factory.generate_envs(**generate_envs)
    e = envs[0]
    e.reset()
    feature = feature_factory(feature_str)

    net = NetFTQ(n_in=len(feature(e.reset(), e)),
                 n_out=e.action_space.n,
                 **ftq_net_params)

    algo = PytorchFittedQ(
        device=device,
        test_policy=None,
        workspace=workspace,
        action_str=None if not hasattr(e, "action_str") else e.action_str,
        policy_network=net,
        gamma=gamma,
        **ftq_params
    )

    import os
    if not os.path.isabs(policy_path):
        actual_policy_path = workspace + "/" + policy_path
    else:
        actual_policy_path = policy_path

    pi = algo.load_policy(policy_path=actual_policy_path)

    pi = PytorchFittedPolicy(pi, e, feature)
    makedirs(path_results)
    set_seed(seed, e)
    _, results = urpy.execute_policy(env=e,
                                     pi=pi,
                                     gamma_r=gamma,
                                     gamma_c=gamma_c,
                                     N_dialogues=N_trajs,
                                     save_path="{}/lambda={}.results".format(path_results, lambda_))

    print("FTQ({}) : {}".format(lambda_, urpy.format_results(results)))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "../config/test_egreedy.json"
    from ncarrara.budgeted_rl.tools.configuration import C

    C.load(config_file).load_pytorch().load_matplotlib('agg')
    main(lambda_=0,
         device=C.device,
         seed=C.seed,
         workspace=C.path_learn_ftq_egreedy,
         path_results = C.path_learn_ftq_egreedy,
         **C.dict["test_ftq"],
         **C.dict)
