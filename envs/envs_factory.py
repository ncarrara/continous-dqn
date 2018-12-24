from sklearn.model_selection import ParameterGrid
import envs.cart_pole_config_env as envsconfig
import gym
import numpy as np

def generate_env_params_combination(env_params):
    for k, v in env_params.items():
        if type(v) is str:
            x = eval(v)
            env_params[k] = x
    grid = ParameterGrid(env_params)
    return grid

def generate_envs(env_str, env_params):
    print(env_params)
    grid = generate_env_params_combination(env_params)
    print("number of envs : {}".format(len(grid)))
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
