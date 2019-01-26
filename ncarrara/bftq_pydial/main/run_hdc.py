# coding=utf-8
from ncarrara.bftq_pydial.tools.configuration import C
from ncarrara.bftq_pydial.tools.policies import HandcraftedSlotFillingEnv
from ncarrara.bftq_pydial.tools.utils_run_pydial import execute_policy, format_results
from ncarrara.utils.os import empty_directory, makedirs
from ncarrara.utils_rl.environments.envs_factory import generate_envs
import logging
import numpy as np


def main(safenesses):
    logger = logging.getLogger(__name__)
    envs, params = generate_envs(**C["generate_envs"])
    empty_directory(C.path_hdc_results)
    e = envs[0]
    makedirs(C.path_hdc_results)

    for safeness in safenesses:
        _, results = execute_policy(e, HandcraftedSlotFillingEnv(e=e, safeness=safeness),
                                    C["gamma"],
                                    C["gamma_c"],
                                    N_dialogues=C["main"]["N_trajs"],
                                    save_path="{}/safeness={}.results".format(C.path_hdc_results, safeness)
                                    )
        print("HDC({:.2f}) : {} ".format(safeness, format_results(results)))


if __name__ == "__main__":
    C.load("config/test_slot_filling.json").load_pytorch()
    main(safeness=[0., 1.])
