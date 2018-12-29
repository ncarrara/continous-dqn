import matplotlib.pyplot as plt
import bftq_pydial.tools.filled_step as fs
import os
import numpy as np
from bftq_pydial.tools.configuration import C

def create_Q_histograms(title, values, path, labels):
    C.makedirs(path)
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
    C.makedirs(path)
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
    # bins=np.linspace(-1.5,1.5,100),
    # for i, value in enumerate(values):
    #     n, bins, patches = plt.hist(x=value, label=labs[i], alpha=1., rwidth=0.5, hatch=hatchs[i % len(hatchs)])  # ,bins=np.linspace(-2,2,100))  # , alpha=0.7, rwidth=0.85)
    # n, bins, patches = plt.hist(x=values,  label=labs, alpha=1.,rwidth=0.5,stacked=True)#,bins=np.linspace(-2,2,100))  # , alpha=0.7, rwidth=0.85)

    edges = np.linspace(-2, 2, 200, endpoint=True)
    hist_func = fs.partial(np.histogram, bins=edges)
    colors = plt.get_cmap('tab10').colors
    #['b', 'g', 'r', 'c', 'm', 'y', 'k']
    hatchs = ['/', '*', '+', '|']
    cols=[]
    hats =[]
    for i,lab in enumerate(labels):
        cols.append(colors[i%len(colors)])
        hats.append(hatchs[i%len(hatchs)])

    color_cycle = fs.cycler(facecolor=cols)
    label_cycle = fs.cycler('label', labs)
    hatch_cycle = fs.cycler('hatch', hats)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    arts = fs.stack_hist(ax, values, color_cycle + label_cycle + hatch_cycle,
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





