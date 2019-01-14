from sklearn.model_selection import ParameterGrid
from ncarrara.utils_rl.environments.lunar_lander_config_env import LunarLanderConfigEnv
import gym
import logging
from gym_pydial.env.env_pydial import EnvPydial
from ncarrara.utils_rl.environments.cart_pole_config_env import CartPoleConfigEnv
from ncarrara.utils_rl.environments.mountain_car_config_env import MountainCarConfigEnv
from ncarrara.continuous_dqn.tools.configuration import C
from ncarrara.utils_rl.environments.slot_filling_env.slot_filling_env import SlotFillingEnv
import numpy as np
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
        if envs_str == CartPoleConfigEnv.ID:
            env = CartPoleConfigEnv(**param)
        elif envs_str == MountainCarConfigEnv.ID:
            env = MountainCarConfigEnv(**param)
        elif envs_str == LunarLanderConfigEnv.ID:
            env = LunarLanderConfigEnv(**param)
        elif envs_str == "gym_pydial":
            env = EnvPydial(seed=C.seed, pydial_logging_level="ERROR", **param)
        elif envs_str == SlotFillingEnv.ID:
            env = SlotFillingEnv(**param)
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
