from ncarrara.bftq_pydial.tools.configuration import C
import sys

print(sys.argv)

if len(sys.argv) > 1:
    C.load_matplotlib('agg').load("config/{}.json".format(sys.argv[1])).create_fresh_workspace()
else:
    C.load("config/yoyo.json")  # .create_fresh_workspace()

from ncarrara.bftq_pydial.main import run_ftq, create_data, run_bftq
import ncarrara.bftq_pydial.main.run_hdc as run_hdc
import ncarrara.bftq_pydial.main.plot_data as plot_data
import numpy as np

create_data.main()

# logging.getLogger("ncarrara.utils_rl.environments.slot_filling_env.slot_filling_env").setLevel("INFO")
run_hdc.main(safenesses=np.linspace(0, 1, 100))
run_ftq.main(lambdas_=np.linspace(0, 1000, 21))
nb_betas = C["nb_betas"]
betas = np.concatenate(
    (np.array([0.]),
     np.exp(np.power(np.linspace(0, nb_betas, nb_betas), np.full(nb_betas, 2. / 3.))) / (
         np.exp(np.power(nb_betas, 2. / 3.)))))

run_bftq.main(betas_test=betas)

# plot_data.main([C.path_bftq_results])
# plot_data.main([C.path_bftq_results])
