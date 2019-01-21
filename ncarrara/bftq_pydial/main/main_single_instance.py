# from ncarrara.bftq_pydial.main.plot_data import main
import torch
from sklearn.model_selection import ParameterGrid
import os
from ncarrara.bftq_pydial.tools.configuration import C
import sys
import numpy as np
import logging
import re

from ncarrara.utils.os import makedirs

if len(sys.argv) > 1:
    config_file = sys.argv[1]
    C.load_matplotlib('agg')
else:
    config_file = "config/test.json"

# C.load_matplotlib('agg')

from ncarrara.bftq_pydial.main import run_ftq, create_data, run_hdc, learn_bftq, test_bftq, plot_data, run_dqn

C.load(config_file).create_fresh_workspace(force=False)

# CREATE DATA DQN or FTQ #
print("learning dqn ...")
run_dqn.main()
torch.cuda.empty_cache()
# create_data.main()
# BFTQ #
print("learning bftq ...")
betas_test = eval(C["betas_test"])
learn_bftq.main()
print("testing bftq ...")
torch.cuda.empty_cache()
test_bftq.main(betas_test=betas_test)
# HDC #
print("testing HDC ...")
run_hdc.main(safenesses=np.linspace(0, 1, 10))
# FTQ #
print("learning and testing FTQ ...")
torch.cuda.empty_cache()
lambdas = eval(C["lambdas"])
run_ftq.main(lambdas_=lambdas, empty_previous_test=True)
# plotting results #
plot_data.main([
    [C.path_bftq_results],
    [C.path_ftq_results],
    [C.path_hdc_results]
])
