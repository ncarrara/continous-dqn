from continuous_dqn.envs.envs_factory import generate_envs
from continuous_dqn.tools import utils as utils
import logging
from continuous_dqn.tools.configuration import C
import json

from utils_rl.transition.replay_memory import ReplayMemory
from utils_rl.transition.transition import TransitionGym


def main():
    logger = logging.getLogger(__name__)
    envs, params = generate_envs(**C["source_envs"])
    seed = C["general"]["seed"]
    N_trajs = C["N_base_trajectories"]
    for ienv, env in enumerate(envs):
        logger.info("generating samples for env {}".format(ienv))
        utils.set_seed(seed=seed, env=env)
        env.seed(seed)
        rm = ReplayMemory(10000,TransitionGym)
        # print(C.CONFIG["N_base_trajectories"],type(C.CONFIG["N_base_trajectories"]))
        for i_traj in range(N_trajs):
            if i_traj % (N_trajs / 10) == 0:
                logger.info("i_traj={}/{}".format(i_traj, N_trajs))
            s = env.reset()
            done = False
            while not done:
                # env.render()
                a = env.action_space.sample()
                s_, r_, done, info = env.step(a)
                rm.push(s.tolist(), a, r_, s_.tolist(), done, info)
                s = s_
        rm.save_memory(C.path_samples + "/{}.json".format(ienv))
    with open(C.path_sources_params, 'w') as file:
        dump = json.dumps(params, indent=4)
        print(dump)
        file.write(dump)


if __name__ == "__main__":
    C.load("config/0_random.json")
    main()
