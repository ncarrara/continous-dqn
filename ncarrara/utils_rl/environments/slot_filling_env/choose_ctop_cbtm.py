import numpy as np
import matplotlib.pyplot as plt


def sucess(cok=1, std=0.2):
    reco_sucess = 1. / (1 + np.exp(-np.random.normal(cok, std)))
    return reco_sucess


def error(cerr=-1, std=0.2):
    reco_err = 1. / (1 + np.exp(-np.random.normal(cerr, std)))
    return reco_err


def main(cbot, ctop, std):
    succ = [sucess(ctop,std) for _ in range(0, 10000)]
    err = [error(cbot,std) for _ in range(0, 10000)]

    plt.hist(succ,500,alpha=0.50)
    plt.hist(err,500,alpha=0.50)
    plt.show()

# main(-1, 1, 0.2)
# main(-1, 1, 0.2)
# main(-1, 1, 0.2)
# main(-0.5, 0.5, 0.2)
# main(-1, 1, 0.6)
