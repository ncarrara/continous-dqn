import logging
import json

from ncarrara.continuous_dqn.envs.envs_factory import generate_envs
from ncarrara.continuous_dqn.tools import utils as utils
from ncarrara.continuous_dqn.tools.configuration import C
from ncarrara.continuous_dqn.tools.features import str_to_feature
from ncarrara.utils.math import set_seed
from ncarrara.utils_rl.transition.replay_memory import ReplayMemory
from ncarrara.utils_rl.transition.transition import TransitionGym
import numpy as np
logger = logging.getLogger(__name__)


def main():
    envs, params = generate_envs(**C["source_envs"])
    seed = C["general"]["seed"]
    feat =str_to_feature(C["feature_str"])
    N_trajs = C["N_base_trajectories"]
    for ienv, env in enumerate(envs):
        logger.info("generating samples for env {}".format(ienv))
        set_seed(seed=seed)
        env.seed(seed)
        rm = ReplayMemory(10000,TransitionGym)
        for i_traj in range(N_trajs):
            if i_traj % (N_trajs / 10) == 0:
                logger.info("i_traj={}/{}".format(i_traj, N_trajs))
            s = env.reset()
            done = False
            while not done:
                if hasattr(env,"action_space_executable"):
                    a = np.random.choice(env.action_space_executable())
                else:
                    a = env.action_space.sample()
                s_, r_, done, info = env.step(a)
                rm.push(feat(s,env), a, r_, feat(s_,env), done, info)
                s = s_
        rm.save_memory(C.path_samples + "/{}.json".format(ienv))
    with open(C.path_sources_params, 'w') as file:
        dump = json.dumps(params, indent=4)
        print(dump)
        file.write(dump)


if __name__ == "__main__":
    C.load("config/0_random.json")
    main()
