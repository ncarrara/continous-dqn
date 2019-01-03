from ncarrara.bftq_pydial.main import compare_ftq_vs_bftq, create_data
from ncarrara.bftq_pydial.tools.configuration import C
from ncarrara.bftq_pydial.tools.policies import HandcraftedSlotFillingEnv
from ncarrara.bftq_pydial.tools.utils_run_pydial import execute_policy, print_results
from ncarrara.utils_rl.environments.envs_factory import generate_envs
import logging
C.load("config/test_slot_filling.json")#.create_fresh_workspace()

# create_data.main()



logging.getLogger("ncarrara.utils_rl.environments.slot_filling_env.slot_filling_env").setLevel("INFO")

for lambda_ in [0]:#>,50,100,1000]:
    compare_ftq_vs_bftq.main(lambda_=lambda_)


logging.getLogger("ncarrara.utils_rl.environments.slot_filling_env.slot_filling_env").setLevel("ERROR")


envs, params = generate_envs(**C["generate_envs"])
e = envs[0]
e.reset()

for safeness in [0., 0.5, 1.0]:
    _, results = execute_policy(e, HandcraftedSlotFillingEnv(e=e, safeness=safeness),
                                     C["gamma"],
                                     C["gamma_c"],
                                     N_dialogues=C["main"]["N_trajs"])
    print("HDC({})".format(safeness))
    print_results(results)