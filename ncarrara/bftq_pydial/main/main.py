# from ncarrara.bftq_pydial.main.plot_data import main
import torch

from ncarrara.bftq_pydial.tools.configuration import C
import sys
import numpy as np
#
if len(sys.argv) > 1:
    C.load_matplotlib('agg').load("config/{}.json".format(sys.argv[1])).create_fresh_workspace()
else:
    C.load("config/16.json")#.create_fresh_workspace()

from ncarrara.bftq_pydial.main import run_ftq, create_data, run_hdc, learn_bftq, test_bftq


# create_data.main()
#
# run_hdc.main(safenesses=np.linspace(0, 1, 10))


betas_test = eval(C["betas_test"])
learn_bftq.main()
torch.cuda.empty_cache()
test_bftq.main(betas_test=betas_test)
torch.cuda.empty_cache()
lambdas = eval(C["lambdas"])
run_ftq.main(lambdas_=lambdas)

# main([
#     "tmp/16/hdc/results",
#     # "tmp/16/ftq/results",
#     "tmp/16/bftq/results",
#
# ])