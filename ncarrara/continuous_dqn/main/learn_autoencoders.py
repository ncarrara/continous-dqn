from ncarrara.continuous_dqn.tools import utils as utils, features
from ncarrara.continuous_dqn.ae.autoencoder import Autoencoder

import os
import logging

from ncarrara.continuous_dqn.tools.configuration import C
from ncarrara.continuous_dqn.tools.features import build_feature_autoencoder
from ncarrara.utils.os import makedirs
from ncarrara.utils.torch_utils import loss_fonction_factory, optimizer_factory

logger = logging.getLogger(__name__)


def main(loss_function_str, optimizer_str, weight_decay, learning_rate, normalize,
         autoencoder_size, n_epochs):
    import torch
    loss_function = loss_fonction_factory(loss_function_str)
    makedirs(C.path_samples)
    feature = build_feature_autoencoder(C["feature_autoencoder_info"])

    all_transitions = utils.read_samples_for_autoencoders(C.path_samples, feature)
    min_n, max_n = autoencoder_size
    autoencoders = [Autoencoder(transitions.shape[1], min_n, max_n,device=C.device) for transitions in all_transitions]
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

        if not os.path.exists(C.path_models):
            os.makedirs(C.path_models)
        path_autoencoder = C.path_models / "{}.pt".format(ienv)
        logger.info("saving autoencoder at {}".format(path_autoencoder))
        torch.save(autoencoders[ienv], path_autoencoder)



