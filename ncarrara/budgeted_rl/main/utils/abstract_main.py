from sklearn.model_selection import ParameterGrid
import os
from ncarrara.budgeted_rl.tools.configuration import C
import sys
import re

from ncarrara.utils.os import makedirs


def main(config_file, override_param_grid, f):
    with open(config_file, 'r') as infile:
        import json

        dict = json.load(infile)
        workspace = dict["general"]["workspace"]
        makedirs(workspace)

    if "matplotlib_backend" in dict["general"]:
        backend = dict["general"]["matplotlib_backend"]
    else:
        backend = "Agg"
    C.load_pytorch().load_matplotlib(backend)

    grid = ParameterGrid(override_param_grid)

    if os.path.exists(workspace + "/params"):
        with open(workspace + "/params", 'r') as infile:
            lines = infile.readlines()
            id_offset = re.match('^id=([0-9]+) ', lines[-1])
        id_offset = int(id_offset.group(1)) + 1
    else:
        id_offset = 0

    str_params = ""
    for i_config, params in enumerate(grid):
        str_params += "id=" + str(id_offset + i_config) + ' ' + ''.join(
            [k + "=" + str(v) + ' ' for k, v in params.items()]) + '\n'

    with open(workspace + "/params", 'a') as infile:
        infile.write(str_params)

    for i_config, params in enumerate(grid):
        i_config = id_offset + i_config
        for k, v in params.items():
            keys = k.split('.')

            tochange = dict
            for ik in range(len(keys) - 1):
                tochange = tochange[keys[ik]]
            tochange[keys[-1]] = v
        dict["general"]["workspace"] = workspace + "/" + str(i_config)
        C.load(dict).create_fresh_workspace(force=True)
        C.dump_to_workspace()

        print("\n-------- i_config={} ----------\n".format(i_config))
        f(C)


if __name__ == "__main__":
    config_file = "config/test_highway.json"

    override_param_grid = {
        'general.seed': [0, 1, 2],
    }


    def f():
        print("f1"),
        print("f2"),
        print("f3"),


    main(config_file, override_param_grid, f)
