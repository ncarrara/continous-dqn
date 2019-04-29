from ncarrara.continuous_dqn.ae.autoencoders import AutoEncoder
from ncarrara.continuous_dqn.tools import utils as utils, features

import os
import logging

from ncarrara.continuous_dqn.tools.features import build_feature_autoencoder
from ncarrara.utils.os import makedirs
from ncarrara.utils.torch_utils import loss_fonction_factory, optimizer_factory

logger = logging.getLogger(__name__)


def main(loss_function_str, optimizer_str, weight_decay, learning_rate, normalize,
         autoencoder_size, n_epochs, feature_autoencoder_info, workspace, device, type_ae="AEA", N_actions=None,
         writer=None):
    import torch
    loss_function = loss_fonction_factory(loss_function_str)
    makedirs(workspace)
    feature = build_feature_autoencoder(feature_autoencoder_info)
    min_n, max_n = autoencoder_size

    all_transitions = utils.read_samples_for_ae(workspace / "samples", feature, N_actions)

    autoencoders = [
        AutoEncoder(n_in=transitions.X.shape[1],
                    n_out=transitions.X.shape[1] * (N_actions if type_ae == "AEA" else 1),
                    min_n=min_n,
                    max_n=max_n,
                    device=device)
        for transitions in all_transitions]

    path_auto_encoders = workspace / "ae"
    makedirs(path_auto_encoders)
    print("learning_rate", learning_rate)
    print("optimizer_str", optimizer_str)
    print("weight_decay", weight_decay)
    # exit()
    for ienv, transitions in enumerate(all_transitions):
        autoencoders[ienv].reset()
        optimizer = optimizer_factory(
            optimizer_str,
            autoencoders[ienv].parameters(),
            lr=learning_rate,
            weight_decay=weight_decay)
        # for x,y in zip(transitions.X,transitions.A):
        #     print(x,"->",y)
        autoencoders[ienv].fit(transitions,
                               size_minibatch=all_transitions[ienv].X.shape[0],
                               n_epochs=n_epochs,
                               optimizer=optimizer,
                               normalize=normalize,
                               stop_loss=0.01,
                               loss_function=loss_function,
                               writer=writer)

        path_autoencoder = path_auto_encoders / "{}.pt".format(ienv)
        logger.info("saving autoencoder at {}".format(path_autoencoder))
        torch.save(autoencoders[ienv], path_autoencoder)
