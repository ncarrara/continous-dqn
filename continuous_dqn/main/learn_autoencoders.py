import torch
from continuous_dqn.tools import utils as utils
from continuous_dqn.ae.autoencoder import Autoencoder
import torch.nn.functional as F
import os
import logging
from continuous_dqn.tools.configuration import C

logger = logging.getLogger(__name__)


def test(criterion, optimizer_str, weight_decay, learning_rate, normalize,
         autoencoder_size, n_epochs):
    criterion = F.l1_loss if criterion == "l1" else None
    C.makedirs(C.path_samples)
    all_transitions = utils.read_samples(C.path_samples)
    min_n, max_n = autoencoder_size
    autoencoders = [Autoencoder(transitions.shape[1], min_n, max_n) for transitions in all_transitions]
    for ienv, transitions in enumerate(all_transitions):
        if optimizer_str == "RMS_PROP":
            optimizer = torch.optim.RMSprop(params=autoencoders[ienv].parameters(),
                                            weight_decay=weight_decay)
        elif optimizer_str == "ADAM":
            optimizer = torch.optim.Adam(params=autoencoders[ienv].parameters(),
                                         lr=learning_rate,
                                         weight_decay=weight_decay)
        else:
            raise Exception("optimizer unknown : {}".format(optimizer_str))
        autoencoders[ienv].fit(transitions,
                               size_minibatch=all_transitions[ienv].shape[0],
                               n_epochs=n_epochs,
                               optimizer=optimizer,
                               normalize=normalize,
                               stop_loss=0.01,
                               criterion=criterion)


        if not os.path.exists(C.path_models):
            os.makedirs(C.path_models)
        path_autoencoder = C.path_models + "/{}.pt".format(ienv)
        logger.info("saving autoencoder at {}".format(path_autoencoder))
        torch.save(autoencoders[ienv], path_autoencoder)


def main():
    test(**C["learn_autoencoders"])


if __name__ == "__main__":
    C.load("config/0_random.json")
    main()
