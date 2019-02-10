# from ncarrara.bftq_pydial.main.plot_data import main
from sklearn.model_selection import ParameterGrid
import os
from ncarrara.bftq_pydial.tools.configuration import C
import sys
import numpy as np
import logging
import re

from ncarrara.utils.os import makedirs
seeds = None
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    if len(sys.argv) >2:
        seed_start = int(sys.argv[2])
        number_seeds = int(sys.argv[3])
        seeds = range(seed_start, seed_start + number_seeds)
else:
    # config_file = "config/test.json"
    config_file = "config/test_highway.json"


with open(config_file, 'r') as infile:
    import json
    dict = json.load(infile)
    workspace = dict["general"]["workspace"]
    makedirs(workspace)

if "matplotlib_backend" in dict["general"]:
    backend = dict["general"]["matplotlib_backend"]
else:
    backend="Agg"
C.load_pytorch().load_matplotlib(backend)

if seeds is None:
    seeds=[dict["general"]["seed"]]

print("seeds = {}".format([str(s) for s in seeds]))

from ncarrara.bftq_pydial.main import run_ftq, create_data, run_hdc, learn_bftq, test_bftq, plot_data,run_dqn

# param_grid = {
#     'general.seed': seeds,
#     'net_params.intra_layers': [[128, 256], [32, 64], [512, 256, 128, 64, 32], [32, 128, 256, 512], [64, 128]],
#     'ftq_params.weight_decay': [0.0, 0.01, 0.001,0.0001],
#     'ftq_params.optimizer': ["RMS_PROP", "ADAM"],
#     'ftq_params.learning_rate': [0.01, 0.001]
# }

# param_grid = {
#     'net_params.intra_layers': [[64,32]],
#     'ftq_params.weight_decay': [0.001],
#     'ftq_params.optimizer': ["ADAM"],
#     'ftq_params.learning_rate': [0.01],
#     'ftq_params.max_nn_epoch':[10000],
#     'ftq_params.reset_policy_each_ftq_epoch':[True,False],
#     'general.seed': seeds,

# }

param_grid = {
    'general.seed': seeds,
}

grid = ParameterGrid(param_grid)

if os.path.exists(workspace + "/params"):
    with open(workspace + "/params", 'r') as infile:
        lines = infile.readlines()
        # print(lines)
        id_offset = re.match('^id=([0-9]+) ', lines[-1])
    id_offset = int(id_offset.group(1))+1
else:
    id_offset=0

str_params = ""
for i_config,params in enumerate(grid):
    str_params += "id=" + str(id_offset + i_config) + ' ' + ''.join([k + "=" + str(v) + ' ' for k, v in params.items()]) + '\n'

with open(workspace + "/params", 'a') as infile:
    infile.write(str_params)

for i_config,params in enumerate(grid):
    i_config = id_offset + i_config
    for k, v in params.items():
        keys = k.split('.')

        tochange = dict
        for ik in range(len(keys) - 1):
            tochange = tochange[keys[ik]]
        tochange[keys[-1]] = v
    dict["general"]["workspace"] = workspace + "/"+ str(i_config)
    C.load(dict).create_fresh_workspace(force=True).load_pytorch()

    print("\n-------- i_config={} ----------\n".format(i_config))
    print("Learning DQN")
    run_dqn.main()

    # print("Learning and testing FTQ")
    # lambdas = eval(C["lambdas"])
    # run_ftq.main(lambdas_=lambdas, empty_previous_test=True)

    print("Learning BFTQ")
    betas_test = eval(C["betas_test"])
    learn_bftq.main()
    print("Testing BFTQ")
    test_bftq.main(betas_test=betas_test)

    # # HDC #
    # print("Testing HDC")
    # run_hdc.main(safenesses=np.linspace(0, 1, 10))
