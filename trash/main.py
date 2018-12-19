from trash.rbm import RBM
import gym
from dqn.replay_memory import ReplayMemory
import torch
import utils
from trash import rbm as utils_rbm
import numpy as np
from sklearn.model_selection import ParameterGrid

np.set_printoptions(precision=2)

print('Training RBM...')

rm = ReplayMemory(10000)

# render = lambda : plt.imshow(env.render(mode='rgb_array'))
env = gym.make('CartPole-v0')

utils.set_seed(seed=0, env=env)

for _ in range(100):
    s = env.reset()
    done = False
    while not done:
        a = env.action_space.sample()
        s_, r_, done, info = env.step(a)  # take a random action
        rm.push(s, a, r_, s_)
        s = s_
print(rm)

rbm_transitions = utils_rbm.transtions_to_rbm_transtions(rm.memory, env.action_space.n)
# print(rbm_transitions)


########## CONFIGURATION ##########
utils.set_device()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CUDA = torch.cuda.is_available()

batch = torch.from_numpy(rbm_transitions).float().to(device)

param_grid = {'num_hidden': [16, 64, 256],
              'k': [1, 2,  5],
              'learning_rate': [0.001, 0.01, 0.1, 1.0],
              'momentum_coefficient': [0.1, 0.5,  0.9],
              'weight_decay': [0.0001, 0.001, 0.01, 0.1],
              'use_cuda':[True]}

grid = ParameterGrid(param_grid)
print(''.join([str(iparam) + " " + str(param) + "\n" for iparam, param in enumerate(grid)]))
min = np.inf
argmin = None
for params in grid:
    rbm = RBM(num_visible=rbm_transitions.shape[1], **params)

    for epoch in range(500):
        epoch_error = 0.0
        batch_error = rbm.contrastive_divergence(batch)
        epoch_error += batch_error
    if epoch_error <min:
        min = epoch_error
        argmin=params
    print(epoch_error,params)
print(min, argmin)

# tensor(3584.2515, device='cuda:1') {'k': 1, 'learning_rate': 0.1, 'momentum_coefficient': 0.9, 'num_hidden': 256, 'use_cuda': True, 'weight_decay': 0.0001}

# print(batch)

# print(rbm.sample_visible(rbm.sample_hidden(batch)))
