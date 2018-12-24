import os
from configuration import C
import torch.nn.functional as F
import utils
import random

import numpy as np

from dqn.dqn import DQN
from dqn.net import Net
import matplotlib.pyplot as plt
from dqn.transfer_module import TransferModule
import logging

logger = logging.getLogger(__name__)

def run_dqn_with_transfer(i_env, env, autoencoders, ers, net_params, dqn_params, decay, N, seed, env_params):
    return run_dqn(i_env, env, autoencoders, ers, net_params=net_params, dqn_params=dqn_params, decay=decay, N=N,
                   seed=seed, env_params=env_params)


def run_dqn_without_transfer(i_env, env, net_params, dqn_params, decay, N, seed, env_params):
    return run_dqn(i_env, env, autoencoders=None, ers=None, net_params=net_params, dqn_params=dqn_params, decay=decay,
                   N=N, seed=seed,
                   env_params=env_params)


def run_dqn(i_env, env, autoencoders, ers, net_params, dqn_params, decay, N, seed, env_params):
    do_transfer = autoencoders is not None or ers is not None
    if do_transfer:
        tm = TransferModule(models=autoencoders,
                            loss=F.l1_loss,
                            experience_replays=ers)
        tm.reset()
    net = Net(**net_params)
    dqn = DQN(policy_network=net, **dqn_params)
    dqn.reset()
    utils.set_seed(seed=seed, env=env)
    env.seed(C.seed)
    # render = lambda: plt.imshow(env.render(mode='rgb_array'))

    rrr = []
    rrr_greedy = []
    epsilons = utils.epsilon_decay(start=1.0, decay=decay, N=N)
    # plt.plot(range(N), epsilons)
    # plt.show()
    # plt.close()
    nb_samples = 0
    for n in range(N):
        s = env.reset()
        done = False
        rr = 0

        while not done:

            if random.random() < epsilons[n]:
                a = env.action_space.sample()
            else:
                a = dqn.pi(s, np.zeros(env.action_space.n))
            s_, r_, done, info = env.step(a)
            rr += r_
            t = (s.tolist(), a, r_, s_.tolist(), done, info)
            dqn.update(*t)
            if do_transfer:
                tm.push(*t)

                best_er, errors = tm.best_source_transitions()

                dqn.update_transfer_experience_replay(best_er)
                if nb_samples < 10:
                    logger.info("[N_trajs={},N_samples={}] errors = {}".format(n, nb_samples,
                                                                         utils.format_errors(errors, i_env)))

            s = s_
            nb_samples += 1
        if do_transfer and n % 50 == 0:
            logger.info("------------------------------------")
            logger.info("[N_trajs={},N_samples={}] errors = {}".format(n, nb_samples, utils.format_errors(errors, i_env)))
            logger.info("Best fit ER's env parameters : {}".format(env_params[np.argmin(errors)]))
            logger.info("Actual env parameters: {}".format(env_params[i_env]))
            logger.info("--------------------------------------")

        rrr.append(rr)

        s = env.reset()
        done = False
        rr = 0
        while not done:
            a = dqn.pi(s, np.zeros(env.action_space.n))
            s_, r_, done, info = env.step(a)
            rr += r_
            s = s_
        rrr_greedy.append(rr)

    return rrr, rrr_greedy
    # n_dots = 10
    # if len(rrr) % int(N / n_dots) == 0 and len(rrr) > 0:
    #     print("n", n)
    #     xxx = np.mean(np.reshape(np.array(rrr), (int(len(rrr) / int(N / n_dots)), -1)), axis=1)
    #     plt.plot(range(len(xxx)), xxx)
    #     plt.show()


"""
=========================
Hatch-filled histograms
=========================

Hatching capabilities for plotting histograms.
"""

import itertools
from collections import OrderedDict
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cycler import cycler
from six.moves import zip


def filled_hist(ax, edges, values, bottoms=None, orientation='v',
                **kwargs):
    """
    Draw a histogram as a stepped patch.

    Extra kwargs are passed through to `fill_between`

    Parameters
    ----------
    ax : Axes
        The axes to plot to

    edges : array
        A length n+1 array giving the left edges of each bin and the
        right edge of the last bin.

    values : array
        A length n array of bin counts or values

    bottoms : scalar or array, optional
        A length n array of the bottom of the bars.  If None, zero is used.

    orientation : {'v', 'h'}
       Orientation of the histogram.  'v' (default) has
       the bars increasing in the positive y-direction.

    Returns
    -------
    ret : PolyCollection
        Artist added to the Axes
    """
    # #print(orientation)
    if orientation not in set('hv'):
        raise ValueError("orientation must be in {{'h', 'v'}} "
                         "not {o}".format(o=orientation))

    kwargs.setdefault('step', 'post')
    edges = np.asarray(edges)
    values = np.asarray(values)
    if len(edges) - 1 != len(values):
        raise ValueError('Must provide one more bin edge than value not: '
                         'len(edges): {lb} len(values): {lv}'.format(
            lb=len(edges), lv=len(values)))

    if bottoms is None:
        bottoms = np.zeros_like(values)
    if np.isscalar(bottoms):
        bottoms = np.ones_like(values) * bottoms

    values = np.r_[values, values[-1]]
    bottoms = np.r_[bottoms, bottoms[-1]]
    if orientation == 'h':
        return ax.fill_betweenx(edges, values, bottoms,
                                **kwargs)
    elif orientation == 'v':
        return ax.fill_between(edges, values, bottoms,
                               **kwargs)
    else:
        raise AssertionError("you should never be here")


def stack_hist(ax, stacked_data, sty_cycle, bottoms=None,
               hist_func=None, labels=None,
               plot_func=None, plot_kwargs=None):
    """
    ax : axes.Axes
        The axes to add artists too

    stacked_data : array or Mapping
        A (N, M) shaped array.  The first dimension will be iterated over to
        compute histograms row-wise

    sty_cycle : Cycler or operable of dict
        Style to apply to each set

    bottoms : array, optional
        The initial positions of the bottoms, defaults to 0

    hist_func : callable, optional
        Must have signature `bin_vals, bin_edges = f(data)`.
        `bin_edges` expected to be one longer than `bin_vals`

    labels : list of str, optional
        The label for each set.

        If not given and stacked data is an array defaults to 'default set {n}'

        If stacked_data is a mapping, and labels is None, default to the keys
        (which may come out in a random order).

        If stacked_data is a mapping and labels is given then only
        the columns listed by be plotted.

    plot_func : callable, optional
        Function to call to draw the histogram must have signature:

          ret = plot_func(ax, edges, top, bottoms=bottoms,
                          label=label, **kwargs)

    plot_kwargs : dict, optional
        Any extra kwargs to pass through to the plotting function.  This
        will be the same for all calls to the plotting function and will
        over-ride the values in cycle.

    Returns
    -------
    arts : dict
        Dictionary of artists keyed on their labels
    """
    # deal with default binning function
    if hist_func is None:
        hist_func = np.histogram

    # deal with default plotting function
    if plot_func is None:
        plot_func = filled_hist

    # deal with default
    if plot_kwargs is None:
        plot_kwargs = {}
    # print(plot_kwargs)
    try:
        l_keys = stacked_data.keys()
        label_data = True
        if labels is None:
            labels = l_keys

    except AttributeError:
        label_data = False
        if labels is None:
            labels = itertools.repeat(None)

    if label_data:
        loop_iter = enumerate((stacked_data[lab], lab, s) for lab, s in
                              zip(labels, sty_cycle))
    else:
        loop_iter = enumerate(zip(stacked_data, labels, sty_cycle))

    arts = {}
    for j, (data, label, sty) in loop_iter:
        if label is None:
            label = 'dflt set {n}'.format(n=j)
        label = sty.pop('label', label)
        vals, edges = hist_func(data)
        if bottoms is None:
            bottoms = np.zeros_like(vals)
        top = bottoms + vals
        # #print(sty)
        sty.update(plot_kwargs)
        # #print(sty)
        ret = plot_func(ax, edges, top, bottoms=bottoms,
                        label=label, **sty)
        bottoms = top
        arts[label] = ret
    ax.legend(fontsize=10)
    return arts


def create_Q_histograms(title, values, path, labels):
    plt.clf()
    maxfreq = 0.
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    n, bins, patches = ax.hist(x=values, label=labels, alpha=1.,
                               stacked=False)  # , bins=np.linspace(-5, 5, 100))  # , alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend(loc='upper right')
    path = path + "/histogram"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + "/" + title)
    plt.close()


def create_Q_histograms_for_actions(title, QQ, path, labels, mask_action=None):
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
    hist_func = partial(np.histogram, bins=edges)
    colors = plt.get_cmap('tab10').colors
    # ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    hatchs = ['/', '*', '+', '|']
    cols = []
    hats = []
    for i, lab in enumerate(labels):
        cols.append(colors[i % len(colors)])
        hats.append(hatchs[i % len(hatchs)])

    color_cycle = cycler(facecolor=cols)
    label_cycle = cycler('label', labs)
    hatch_cycle = cycler('hatch', hats)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    arts = stack_hist(ax, values, color_cycle + label_cycle + hatch_cycle,
                      hist_func=hist_func, labels=labs)

    plt.grid(axis='y', alpha=0.75)

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    path = path + "/histogram"
    # plt.legend()
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + "/" + title)
    plt.close()
