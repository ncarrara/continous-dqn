from dqn.replay_memory import ReplayMemory
import torch
import utils
import numpy as np
from ae.autoencoder import Autoencoder
from envs.envs_factory import generate_envs
import torch.nn.functional as F
import configuration as config
from sklearn.model_selection import ParameterGrid

envs = []
params = []

envs, params = generate_envs("CartPoleConfig-v0", {"gravity": [9.8],  # [9.8],
                                                   "masscart": [1.0],
                                                   "masspole": [0.1],
                                                   "length": [0.01,0.5,1.0],#np.concatenate((np.array([0.01, 0.02]), np.linspace(0.5, 10, 21))),
                                                   "force_mag": [10.0],
                                                   })

gridsearch = {
    "criterion": [F.l1_loss],#, F.mse_loss, F.smooth_l1_loss],
    "optimizer_str": ["ADAM"],#"["RMS_PROP", "ADAM"],
    "weight_decay": [0.0],
    "learning_rate": [0.1],#, 0.01],
    "normalize": [True],#, False],
    "autoencoder_size": [(2,16)],#[(2, 128), (8, 32)]
    "n_epochs":[1000]
}

all_transitions = []

for ienv, env in enumerate(envs):
    utils.set_seed(seed=0, env=env)
    rm = ReplayMemory(10000)
    for _ in range(1000):
        s = env.reset()
        done = False
        while not done:
            # env.render()
            a = env.action_space.sample()
            s_, r_, done, info = env.step(a)
            rm.push(s, a, r_, s_)
            s = s_
    data = rm.sample_to_numpy(len(rm))
    all_transitions.append(torch.from_numpy(data).float().to(config.DEVICE))


def test(criterion, optimizer_str, weight_decay, learning_rate, normalize, autoencoder_size,n_epochs):
    min_n, max_n = autoencoder_size
    autoencoders = [Autoencoder(all_transitions[ienv].shape[1], min_n, max_n) for ienv in range(len(envs))]

    for ienv in range(len(envs)):
        if optimizer_str == "RMS_PROP":
            optimizer = torch.optim.RMSprop(params=autoencoders[ienv].parameters(),
                                            weight_decay=weight_decay)
        elif optimizer_str == "ADAM":
            optimizer = torch.optim.Adam(params=autoencoders[ienv].parameters(),
                                         lr=learning_rate,
                                         weight_decay=weight_decay)
        else:
            raise Exception("optimizer unknown : {}".format(optimizer_str))

        autoencoders[ienv].fit(all_transitions[ienv],
                               size_minibatch=all_transitions[ienv].shape[0],
                               n_epochs=n_epochs,
                               optimizer=optimizer,
                               normalize=normalize,
                               stop_loss=0.01,
                               criterion=criterion)

    rez = []
    for ienv in range(len(envs)):
        rez_env = []
        for ienv2 in range(len(envs)):
            x = all_transitions[ienv2]
            loss = criterion(autoencoders[ienv](x), x).item()  # loss symetrique : OK
            rez_env.append(loss)
        rez.append(rez_env)

    print(utils.array_to_cross_comparaison(rez))


grid = ParameterGrid(gridsearch)
for param in grid:
    print(param)
    test(**param)
