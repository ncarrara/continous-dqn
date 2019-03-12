from ncarrara.continuous_dqn.tools import utils as utils, features
from ncarrara.continuous_dqn.ae.autoencoder import Autoencoder

import os
import logging

from ncarrara.continuous_dqn.tools.features import build_feature_autoencoder
from ncarrara.utils.os import makedirs
from ncarrara.utils.torch_utils import loss_fonction_factory, optimizer_factory

logger = logging.getLogger(__name__)


def main(loss_function_str, optimizer_str, weight_decay, learning_rate, normalize,
         autoencoder_size, n_epochs, feature_autoencoder_info, workspace, device):
    import torch
    loss_function = loss_fonction_factory(loss_function_str)
    makedirs(workspace)
    feature = build_feature_autoencoder(feature_autoencoder_info)

    all_transitions = utils.read_samples_for_autoencoders(workspace / "samples_random", feature)


    min_n, max_n = autoencoder_size
    print(all_transitions[0].shape)
    autoencoders = [Autoencoder(transitions.shape[1], min_n, max_n, device=device) for transitions in all_transitions]
    path_auto_encoders = workspace / "ae"
    makedirs(path_auto_encoders)
    for ienv, transitions in enumerate(all_transitions):
        optimizer = optimizer_factory(
            optimizer_str,
            autoencoders[ienv].parameters(),
            lr=learning_rate,
            weight_decay=weight_decay)
        autoencoders[ienv].fit(transitions,
                               size_minibatch=all_transitions[ienv].shape[0],
                               n_epochs=n_epochs,
                               optimizer=optimizer,
                               normalize=normalize,
                               stop_loss=0.01,
                               loss_function=loss_function)

        path_autoencoder = path_auto_encoders / "{}.pt".format(ienv)
        logger.info("saving autoencoder at {}".format(path_autoencoder))
        torch.save(autoencoders[ienv], path_autoencoder)
