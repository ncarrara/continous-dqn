from ncarrara.budgeted_rl.main.utils import test_bftq, test_ftq, abstract_main
from ncarrara.budgeted_rl.main.egreedy import learn_ftq_egreedy, learn_bftq_egreedy
import sys

from ncarrara.budgeted_rl.tools.configuration import C

seeds = None
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    if len(sys.argv) > 2:
        seed_start = int(sys.argv[2])
        number_seeds = int(sys.argv[3])
        seeds = range(seed_start, seed_start + number_seeds)
else:
    config_file = "../config/test_egreedy.json"
    C.load(config_file).create_fresh_workspace(force=True)
    seeds = [0,1]

def main(config):
    lambdas = config.dict["lambdas"]
    if type(lambdas) is str:
        import numpy as np
        lambdas=eval(lambdas)
    for lambda_ in lambdas:
        workspace = config.path_learn_ftq_egreedy + "_lambda={}".format(lambda_)
        learn_ftq_egreedy.main(
            lambda_=lambda_, seed=config.seed, device=config.device,
            workspace=workspace,
            **config.dict["learn_ftq_egreedy"], **config.dict
        )
    for lambda_ in lambdas:
        workspace = config.path_learn_ftq_egreedy + "_lambda={}".format(lambda_)
        test_ftq.main(
            lambda_=lambda_, device=config.device, seed=config.seed,
            workspace=workspace,
            path_results=config.path_ftq_results,
            **config.dict["test_ftq"], **config.dict
        )

    learn_bftq_egreedy.main(
        device=config.device, seed=config.seed,
        workspace=config.path_learn_bftq_egreedy,
        **config.dict["learn_bftq_egreedy"],
        **config.dict
    )
    test_bftq.main(
        device=config.device, seed=config.seed,
        workspace=config.path_learn_bftq_egreedy,
        path_results=config.path_bftq_results,
        **config.dict["test_bftq"], **config.dict
    )


override_param_grid = {
    'general.seed': seeds,
}

abstract_main.main(config_file, override_param_grid, main)

# algos = {
#     "ftq": [[1, 0, 0], r"ftq($\lambda$)"],
#     "bftq": [[0, 1, 0], r"bftq($\beta$)"],
#     # "hdc": [[0, 0, 1], "hdc(safeness)"],
#     # "dqn": [[1, 1, 0], "dqn"]
# }
# plot_all(parse_data(workspace, algos), workspace, algos)
