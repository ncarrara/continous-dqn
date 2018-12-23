import torch
import utils as utils
from ae.autoencoder import Autoencoder
import torch.nn.functional as F
import os
import logging
from configuration import C

logger = logging.getLogger(__name__)


def test(criterion, optimizer_str, weight_decay, learning_rate, normalize,
         autoencoder_size, n_epochs):
    criterion = F.l1_loss if criterion == "l1" else None
    folder_samples = C.workspace+"/"+C.path_samples
    if not os.path.exists(folder_samples):
        os.makedirs(folder_samples)

    all_transitions = utils.read_samples(folder_samples)
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

        folder_models = C.workspace + "/" + C.path_models

        if not os.path.exists(folder_models):
            os.makedirs(folder_models)
        path_autoencoder = folder_models + "/{}.pt".format(ienv)
        logger.info("saving autoencoder at {}".format(path_autoencoder))
        torch.save(autoencoders[ienv], path_autoencoder)


def main():
    test(**C["learn_autoencoders"])


if __name__ == "__main__":
    C.load("config/0.json")
    main()
