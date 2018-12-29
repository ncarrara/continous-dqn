from sklearn.model_selection import ParameterGrid
import continuous_dqn.envs.cart_pole_config_env as cart_pole_config_env
from continuous_dqn import envs as mountain_car_config_env
import continuous_dqn.envs.lunar_lander_config_env as lunar_lander_config_env
import gym
import logging
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
    logger.info("[generate_envs] params : \n{}".format(envs_params))
    grid = generate_env_params_combination(envs_params)
    logger.info("[generate_envs] number of envs : {}".format(len(grid)))

    envs = []
    rez_params = []
    for param in grid:
        if envs_str == cart_pole_config_env.CartPoleConfigEnv.ID:
            env = cart_pole_config_env.CartPoleConfigEnv(**param)
        elif envs_str == mountain_car_config_env.MountainCarConfigEnv.ID:
            env = mountain_car_config_env.MountainCarConfigEnv(**param)
        elif envs_str == lunar_lander_config_env.LunarLanderConfigEnv.ID:
            env = lunar_lander_config_env.LunarLanderConfigEnv(**param)
        else:
            env = gym.make(envs_str)
            for k, v in param.items():
                setattr(env, k, v)
        envs.append(env)
        rez_params.append(param)
    logger.info("[generate_envs] actual params : \n{}".format("".join([str(pa)+"\n" for pa in rez_params])))
    return envs, rez_params
