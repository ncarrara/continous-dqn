from ncarrara.bftq_pydial.tools.configuration import C
import sys

print(sys.argv)

# if len(sys.argv) > 1:
#     C.load_matplotlib('agg').load("config/{}.json".format(sys.argv[1])).create_fresh_workspace()
# else:
#     C.load("config/yaya8.json").create_fresh_workspace()

from ncarrara.bftq_pydial.main import run_ftq, create_data, run_hdc, learn_bftq, test_bftq
import ncarrara.bftq_pydial.main.plot_data as plot_data
import logging
import numpy as np

# create_data.main()


# run_hdc.main(safenesses=np.linspace(0, 1, 50))


# lambdas = eval(C["lambdas"])
# betas_test = eval(C["betas_test"])
# learn_bftq.main()
# test_bftq.main(betas_test=betas_test)
# run_ftq.main(lambdas_=lambdas)



# cf changement de seed abusé
plot_data.main([
    # "tmp/yaya5/bftq/results",
    # "tmp/yaya6/bftq/results",
    # "tmp/yaya7/bftq/results",
    # "tmp/yaya7.1/bftq/results",
    # "tmp/yaya4/ftq/results",
    # "tmp/yaya7.1/ftq/results",
    "tmp/yaya7/bftq/results",
    "tmp/yaya7/ftq/results",
    # "tmp/yaya4/hdc/results",
    # "tmp/yaya7/ftq/results"
])
