# from ncarrara.bftq_pydial.main.plot_data import main
import torch

from ncarrara.bftq_pydial.tools.configuration import C
import sys
import numpy as np
import logging

if len(sys.argv) > 1:
    config_file = sys.argv[1]
    seed_start= int(sys.argv[2])
    number_seeds= int(sys.argv[3])
    seeds = range(seed_start,seed_start+number_seeds)
    C.load_matplotlib('agg')
else:
    config_file = "config/final.json"
    seeds = [0]

# logging.getLogger("ncarrara.bftq_pydial.main.create_data").setLevel(logging.INFO)
# logging.getLogger("ncarrara.utils_rl.environments.slot_filling_env.slot_filling_env").setLevel(logging.INFO)

print("seeds = {}".format(seeds))

from ncarrara.bftq_pydial.main import run_ftq, create_data, run_hdc, learn_bftq, test_bftq, plot_data

with open(config_file, 'r') as infile:
    import json
    dict = json.load(infile)
print(json.dumps(dict, indent=4, sort_keys=True))

for seed in seeds:
    C.load(dict,seed).create_fresh_workspace(force=True)
    create_data.main()
    torch.cuda.empty_cache()
    # run_hdc.main(safenesses=np.linspace(0, 1, 10))
    # betas_test = eval(C["betas_test"])
    # learn_bftq.main()
    # torch.cuda.empty_cache()
    # test_bftq.main(betas_test=betas_test)
    # torch.cuda.empty_cache()
    lambdas = eval(C["lambdas"])
    run_ftq.main(lambdas_=lambdas)
    torch.cuda.empty_cache()
    plot_data.main([["final2/seed=0/ftq/results"]])