from sklearn.model_selection import ParameterGrid
import gym
import logging

from ncarrara.utils_rl.environments.gridworld.world import World

logger = logging.getLogger(__name__)


def generate_env_params_combination(env_params):
    for k, v in env_params.items():
        if type(v) is str:
            x = eval(v)
            env_params[k] = x
            logger.info("[gepc] <{}> {} -> {}".format(k, v, x))
    grid = ParameterGrid(env_params)
    return grid


def generate_envs(envs_str, envs_params):
    logger.info(
        "[generate_envs] params : \n\n{}".format("".join(["\t{} : {}\n".format(k, v) for k, v in envs_params.items()])))
    grid = generate_env_params_combination(envs_params)
    logger.info("[generate_envs] number of envs : {}".format(len(grid)))

    envs = []
    rez_params = []
    for param in grid:
        if envs_str == "CartPoleConfig-v0":
            from ncarrara.utils_rl.environments.cart_pole_config_env import CartPoleConfigEnv
            env = CartPoleConfigEnv(**param)
        elif envs_str == "MountainCarConfig-v0":
            from ncarrara.utils_rl.environments.mountain_car_config_env import MountainCarConfigEnv
            env = MountainCarConfigEnv(**param)
        elif envs_str == "LunarLanderConfig-v0":
            from ncarrara.utils_rl.environments.lunar_lander_config_env import LunarLanderConfigEnv
            env = LunarLanderConfigEnv(**param)
        elif envs_str == "gym_pydial":
            env_pydial = __import__("gym_pydial.env.env_pydial")
            env = env_pydial.EnvPydial(seed=C.seed, pydial_logging_level="ERROR", **param)
        elif envs_str == "slot_filling_env_v0":
            from ncarrara.utils_rl.environments.slot_filling_env.slot_filling_env import SlotFillingEnv
            env = SlotFillingEnv(**param)
        elif envs_str == "highway-v0":
            __import__("highway_env")
            env = gym.make(envs_str)
            env.configure(dict(**param))
        elif envs_str == "test_death_trap":
            from ncarrara.utils_rl.environments.gridworld.model_generator import generate_test_death_trap
            env,_ = generate_test_death_trap()
        elif envs_str == "3xWidth":
            from ncarrara.utils_rl.environments.gridworld.model_generator import generate_3xWidth
            env,_ = generate_3xWidth(**param)
            w = World(env)
            w.draw_frame()
            w.draw_lattice()
            w.draw_cases()
            # beurk
            from ncarrara.budgeted_rl.tools.configuration_bftq import C
            if hasattr(C,"workspace"):
                w.save(C.workspace+"/"+"env")
        else:
            env = gym.make(envs_str)
            for k, v in param.items():
                setattr(env, k, v)
        envs.append(env)
        rez_params.append(param)
    logger.info("[generate_envs] actual params : \n\n{}".format("".join(["{}\n".format("".join(["\t{} : {}\n".format(k, v) for k, v in param.items()])) for param in rez_params])))
    # for param in rez_params:
    #     logger.info("".join(["\t{} : {}\n".format(k, v) for k, v in param.items()]))
    return envs, rez_params
