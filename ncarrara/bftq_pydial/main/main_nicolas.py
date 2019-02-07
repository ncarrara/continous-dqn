from ncarrara.bftq_pydial.main import test_bftq
import numpy as np

from ncarrara.bftq_pydial.tools.configuration import C
from ncarrara.utils.os import makedirs

config_file = "config/egreedy.json"

with open(config_file, 'r') as infile:
    import json

    dict = json.load(infile)
    workspace = dict["general"]["workspace"]
    makedirs(workspace)

if "matplotlib_backend" in dict["general"]:
    backend = dict["general"]["matplotlib_backend"]
else:
    backend = "Agg"

C.load_pytorch().load_matplotlib(backend).load(dict)

test_bftq.main(betas_test=np.linspace(0, 1, 21),
               policy_basename="policy",
               policy_path="/home/sequel/ncarrara/phd_code/ncarrara/bftq_pydial/main/tmp/"
                           "egreedy/bftq/learn_bftq_egreedy/batch=3")
