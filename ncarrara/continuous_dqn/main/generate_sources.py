import logging
import json

from ncarrara.continuous_dqn.dqn.utils_dqn import run_dqn
from ncarrara.continuous_dqn.tools.features import build_feature_dqn
from ncarrara.utils_rl.environments.envs_factory import generate_envs
from ncarrara.utils.math_utils import set_seed
from ncarrara.utils_rl.transition.replay_memory import Memory
import numpy as np
logger = logging.getLogger(__name__)


def main(source_envs, feature_dqn_info, net_params, dqn_params,
         N_dqn,N_random, seed, device, workspace, decay, start_decay, traj_max_size, gamma, writer=None):
    envs, params = generate_envs(**source_envs)

    for ienv, env in enumerate(envs):
        logger.info("generating samples for env {}".format(ienv))

        if N_dqn is not None:
            logger.info("dqn ...".format(ienv))

            set_seed(seed=seed, env=env)
            feature_dqn = build_feature_dqn(feature_dqn_info)
            _, _, memory, dqn = run_dqn(
                env,
                workspace=workspace / "dqn_workspace",
                seed=seed,
                feature_dqn=feature_dqn,
                device=device,
                net_params=net_params,
                dqn_params=dqn_params,
                N=N_dqn,
                decay=decay,
                start_decay=start_decay,
                traj_max_size=traj_max_size,
                gamma=gamma,
                writer=writer)
            memory.save(workspace / "samples_dqn" / "{}.json".format(ienv), as_json=False)
            dqn.save(workspace / "models_dqn" / "{}.pt".format(ienv))

        if N_random is not None:
            memory_random = Memory()
            logger.info("samples for autoencoders (random trajectories) ...".format(ienv))
            for n in range(N_random):
                s = env.reset()
                done = False
                it = 0
                while (not done):

                    if hasattr(env, "action_space_executable"):
                        a = np.random.choice(env.action_space_executable())
                    else:
                        a = env.action_space.sample()
                    s_, r_, done, info = env.step(a)
                    done = done or (traj_max_size is not None and it >= traj_max_size - 1)
                    memory_random.push(s, a, r_, s_, done, info)
                    s = s_
                    it += 1

            memory_random.save(workspace / "samples_random" / "{}.json".format(ienv), as_json=False)

    with open(workspace / 'params.json', 'w') as file:
        dump = json.dumps(params, indent=4)
        print(dump)
        file.write(dump)

# if __name__ == "__main__":
#     from ncarrara.continuous_dqn.tools.configuration import C
#
#     C.load("config/0_pydial.json").load_pytorch()
#     main()
