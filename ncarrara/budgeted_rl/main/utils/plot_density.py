import sys
from pathlib import Path
from json_tricks import dump, dumps, load, loads, strip_comments
from ncarrara.utils_rl.environments.gridworld.world import World


def main(workspace):
    workspace = Path(workspace)
    with open(workspace / 'trajectories.json') as f:
        trajs = load(f)


    selected_trajs = []
    for traj in trajs:
        for sample in traj:
            s,a,r,s,i,e = sample
            if s[1] > 3:
                selected_trajs.append(traj)
                break
    World.plot_density(selected_trajs, workspace / "densities_post.png")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError("Usage: plot_data.py <path>")
    main(sys.argv[1])
