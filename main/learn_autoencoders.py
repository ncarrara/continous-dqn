import torch
import utils as utils
from ae.autoencoder import Autoencoder
import torch.nn.functional as F
import os
import logging
import configuration as c

logger = logging.getLogger(__name__)


def test(path_data, path_models, criterion, optimizer_str, weight_decay, learning_rate, normalize,
         autoencoder_size, n_epochs):
    criterion = F.l1_loss if criterion == "l1" else None
    if not os.path.exists(path_models):
        os.makedirs(path_models)

    all_transitions = utils.read_samples(path_data)
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
        path_autoencoder = path_models + "/{}.pt".format(ienv)
        logger.info("saving autoencoder at {}".format(path_autoencoder))
        torch.save(autoencoders[ienv], path_autoencoder)


def main():
    test(c.C.CONFIG["general"]["path_samples"],
         c.C.CONFIG["general"]["path_models"],
         **c.C.CONFIG["learn_autoencoders"])


if __name__ == "__main__":
    main()
