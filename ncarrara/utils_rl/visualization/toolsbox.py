from cycler import cycler
from functools import partial

import matplotlib.pyplot as plt

from ncarrara import utils_rl as fs
import os
import numpy as np

from ncarrara.utils.os import makedirs
from ncarrara.utils_rl.visualization.filled_step import stack_hist


def create_Q_histograms(title, values, path, labels):
    makedirs(path)
    plt.clf()
    maxfreq = 0.
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    n, bins, patches = ax.hist(x=values, label=labels, alpha=1.,
                                stacked=False, bins=np.linspace(-5, 5, 100))  # , alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend(loc='upper right')
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(path + "/" + title)
    plt.close()

def create_Q_histograms_for_actions(title, QQ, path, labels, mask_action=None):
    makedirs(path)
    #
    # # set up style cycles


    if mask_action is None:
        mask_action = np.zeros(len(QQ[0]))
    labs = []
    for i, label in enumerate(labels):
        # labels[i] = label[0:6]
        if mask_action[i] != 1:
            # labs.append(label[0:6])
            labs.append(label)
    Nact = len(QQ[0])
    values = []
    for act in range(Nact):
        if mask_action[act] != 1:
            value = []
            for i in range(len(QQ)):
                value.append(QQ[i][act])
            values.append(value)
    plt.clf()

    edges = np.linspace(-2, 2, 200, endpoint=True)
    hist_func = partial(np.histogram, bins=edges)
    colors = plt.get_cmap('tab10').colors
    #['b', 'g', 'r', 'c', 'm', 'y', 'k']
    hatchs = ['/', '*', '+', '|']
    cols=[]
    hats =[]
    for i,lab in enumerate(labels):
        cols.append(colors[i%len(colors)])
        hats.append(hatchs[i%len(hatchs)])

    color_cycle = cycler(facecolor=cols)
    label_cycle = cycler('label', labs)
    hatch_cycle = cycler('hatch', hats)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    arts = stack_hist(ax, values, color_cycle + label_cycle + hatch_cycle,
                      hist_func=hist_func,labels=labs)


    plt.grid(axis='y', alpha=0.75)

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend()
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(path + "/" + title)
    plt.close()





