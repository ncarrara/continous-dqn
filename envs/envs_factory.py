from sklearn.model_selection import ParameterGrid
import gym

from gym.envs.classic_control import CartPoleEnv

def generate_envs(env_str, all_params):
    grid = ParameterGrid(all_params)
    envs = []
    rez_params = []
    for param in grid:
        env = gym.make(env_str)
        for k,v in param.items():
            setattr(env,k,v)
        envs.append(env)
        rez_params.append(param)
    return envs,rez_params
