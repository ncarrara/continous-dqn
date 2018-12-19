from dqn.replay_memory import ReplayMemory
import torch
import utils
import numpy as np
from ae.autoencoder import Autoencoder
from envs.envs_factory import generate_envs
import logging
from torch import nn
import torch.nn.functional as F
import gym

print(torch.__version__)

utils.set_device()

logging.basicConfig(level=logging.INFO)
np.set_printoptions(precision=2)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

envs = []
params=[]
#
# envs, params = generate_envs("CartPole-v0", {"gravity": [9.8],
#                                              "masscart": [1.0],
#                                              "masspole": [0.1],
#                                              "length": [0.5],
#                                              "force_mag": [10.0],
#                                              })

envs, params = generate_envs("CartPole-v0", {"gravity": np.linspace(9.8, 20., 2),  # [9.8],
                                             "masscart": [0.0,1.0],
                                             "masspole": [0.1],
                                             "length": [0.5,1.0],
                                             "force_mag": [0.0,10.0],
                                             })

# envs.append(gym.make("MountainCar-v0"))
# params.append("MountainCar-v0")
all_transitions = []

for ienv, env in enumerate(envs):
    # print(params[ienv])
    utils.set_seed(seed=0, env=env)
    rm = ReplayMemory(10000)
    for _ in range(100):
        s = env.reset()
        done = False
        while not done:
            # env.render()
            a = env.action_space.sample()
            s_, r_, done, info = env.step(a)
            rm.push(s, a, r_, s_)
            s = s_
    all_transitions.append(rm.sample_to_tensors(len(rm)).to(device))
    print(all_transitions[ienv].shape[1])
# print(all_transitions)

criterion = F.mse_loss

autoencoders = [Autoencoder(all_transitions[ienv].shape[1], criterion).to(device) for ienv in range(len(envs))]

for ienv in range(len(envs)):
    autoencoders[ienv].fit(all_transitions[ienv])




for ienv in range(len(envs)):
    print("---------------")
    print("autoencoder={}".format(ienv))
    for ienv2 in range(len(envs)):
        x = all_transitions[ienv2]
        print(ienv2, "loss", criterion(autoencoders[ienv](x),x).data)
