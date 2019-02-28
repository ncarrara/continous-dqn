import logging
import json

from ncarrara.utils_rl.environments.envs_factory import generate_envs
from ncarrara.continuous_dqn.tools.configuration import C
from ncarrara.utils.math_utils import set_seed
from ncarrara.utils_rl.transition.replay_memory import Memory
import numpy as np

logger = logging.getLogger(__name__)


def main():
    envs, params = generate_envs(**C["source_envs"])
    seed = C["general"]["seed"]

    N_trajs = C["N_base_trajectories"]
    for ienv, env in enumerate(envs):
        # print(env.action_space.n)
        # print(env.action_space_str)
        logger.info("generating samples for env {}".format(ienv))
        set_seed(seed=seed, env=env)
        rm = Memory()
        for i_traj in range(N_trajs):
            if i_traj % (N_trajs / 10) == 0:
                logger.info("i_traj={}/{}".format(i_traj, N_trajs))
            s = env.reset()

            done = False
            while not done:
                if hasattr(env, "action_space_executable"):
                    a = np.random.choice(env.action_space_executable())
                else:
                    a = env.action_space.sample()
                s_, r_, done, info = env.step(a)

                rm.push(s.tolist() if type(s) == type(np.zeros(1)) else s,
                        int(a),
                        r_,
                        s_.tolist() if type(s_) == type(np.zeros(1)) else s_,
                        done,
                        info)
                s = s_
        rm.save_memory(C.path_samples / "{}.json".format(ienv), C["generate_samples"]["as_json"])
    with open(C.path_sources_params, 'w') as file:
        dump = json.dumps(params, indent=4)
        print(dump)
        file.write(dump)


if __name__ == "__main__":
    C.load("config/0_pydial.json").load_pytorch()
    main()
