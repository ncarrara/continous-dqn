# from ncarrara.bftq_pydial.main.plot_data import main
import torch

from ncarrara.bftq_pydial.tools.configuration import C
import sys
import numpy as np
import logging

if len(sys.argv) > 1:
    C.load_matplotlib('agg').load("config/{}.json".format(sys.argv[1])).create_fresh_workspace()
else:
    C.load("config/23.0.json")#.create_fresh_workspace()

from ncarrara.bftq_pydial.main import run_ftq, create_data, run_hdc, learn_bftq, test_bftq, plot_data

# create_data.main()
#
# run_hdc.main(safenesses=np.linspace(0, 1, 10))
# logging.getLogger("ncarrara.utils_rl.environments.slot_filling_env.slot_filling_env").setLevel(logging.INFO)

# run_hdc.main(safenesses=[0.])


betas_test = eval(C["betas_test"])
learn_bftq.main()
# torch.cuda.empty_cache()

test_bftq.main(betas_test=betas_test)
# torch.cuda.empty_cache()
# lambdas = eval(C["lambdas"])
# run_ftq.main(lambdas_=lambdas)

# plot_data.main([
#     # "tmp/16/hdc/results",
#     # "tmp/16/ftq/results",
#     "tmp/23.0/bftq/results",
#
# ])