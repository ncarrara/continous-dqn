from sklearn.model_selection import ParameterGrid
import envs.cart_pole_config_env as envsconfig
import gym


def generate_envs(env_str, all_params):
    grid = ParameterGrid(all_params)
    envs = []
    rez_params = []
    for param in grid:
        if env_str==envsconfig.CartPoleConfigEnv.ID:
            env=envsconfig.CartPoleConfigEnv(**param)
        else:
            env = gym.make(env_str)
            for k,v in param.items():
                setattr(env,k,v)
        envs.append(env)
        rez_params.append(param)
    return envs,rez_params
