from dqn.replay_memory import ReplayMemory
import utils as utils
from envs.envs_factory import generate_envs
import logging
from configuration import C


def main():
    logger = logging.getLogger(__name__)
    envs, params = generate_envs(**C.CONFIG["generate_samples"])
    seed = C.CONFIG["general"]["seed"]
    N_trajs = C.CONFIG["N_base_trajectories"]
    for ienv, env in enumerate(envs):
        logger.info("generating samples for env {}".format(ienv))
        utils.set_seed(seed=seed, env=env)
        env.seed(seed)
        rm = ReplayMemory(10000)
        # print(C.CONFIG["N_base_trajectories"],type(C.CONFIG["N_base_trajectories"]))
        for i_traj in range(N_trajs):
            if i_traj%(N_trajs/10)==0:
                logger.info("i_traj={}/{}".format(i_traj,N_trajs))
            s = env.reset()
            done = False
            while not done:
                # env.render()
                a = env.action_space.sample()
                s_, r_, done, info = env.step(a)
                rm.push(s.tolist(), a, r_, s_.tolist())
                s = s_
        rm.save_memory(C.workspace + "/" + C.path_samples + "/{}.json".format(ienv))


if __name__ == "__main__":
    main()
