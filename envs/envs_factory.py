from sklearn.model_selection import ParameterGrid
import envs.cart_pole_config_env as envsconfig
import gym
import numpy as np


def generate_envs(env_str, env_params):
    print(env_params)
    for k, v in env_params.items():
        if type(v) is str:
            x = eval(v)
            print(k, x)
            env_params[k] = x
    grid = ParameterGrid(env_params)

    print("nb envs : {}".format(len(grid)))
    envs = []
    rez_params = []
    for param in grid:
        if env_str == envsconfig.CartPoleConfigEnv.ID:
            env = envsconfig.CartPoleConfigEnv(**param)
        else:
            env = gym.make(env_str)
            for k, v in param.items():
                setattr(env, k, v)
        envs.append(env)
        rez_params.append(param)
    return envs, rez_params
