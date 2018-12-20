import torch
import utils
import torch.nn.functional as F
import os
import logging
import configuration as c

def main():
    logger = logging.getLogger(__name__)
    criterion = F.l1_loss

    path_data = c.C.CONFIG["general"]["path_samples"]

    # READING SAMPLES #
    all_transitions = utils.read_samples(path_data)

    # LOADING AUTOENCODERS #
    path_autoencoders = c.C.CONFIG["general"]["path_models"]
    logger.info("reading autoencoders at {}".format(path_autoencoders))
    files_autoencoders = os.listdir(path_autoencoders)
    n_autoencoders = len(files_autoencoders)
    autoencoders = [None] * len(files_autoencoders)
    for file in files_autoencoders:
        i_autoencoder = int(file.split(".")[0])
        path_autoencoder = path_autoencoders + "/" + file
        autoencoders[i_autoencoder] = torch.load(path_autoencoder,map_location=c.DEVICE)

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
