import torch
import utils
import torch.nn.functional as F
import os
import logging
from configuration import C


def main():
    logger = logging.getLogger(__name__)
    criterion = F.l1_loss

    # READING SAMPLES #
    all_transitions = utils.read_samples(C.workspace+"/"+C.path_samples)

    # LOADING AUTOENCODERS #
    autoencoders = utils.load_autoencoders(C.workspace+"/"+C.path_models)

    # WHAT AUTOENCODER REBUILD THE BEST EACH SAMPLES SET ?
    rez = []
    for transitions in all_transitions:
        rez_ae = []
        for ae in autoencoders:
            loss = criterion(ae(transitions), transitions).item()
            rez_ae.append(loss)
        rez.append(rez_ae)

    print(utils.array_to_cross_comparaison(rez))


if __name__ == "__main__":
    # execute only if run as a script
    main()
