import itertools
import sys
import numpy as np
import os

import logging
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_data(path, params):
    with open(path + "/" + "params") as f:
        ids = re.findall(r"id=(\d+)", f.read(), re.MULTILINE)

    # Extract results for each valid (algorithm, id) pair
    data = pd.DataFrame()
    for algo in params.keys():
        results = pd.DataFrame()
        for id in ids:
            # Results directory
            file_id = "{}/{}/{}/results".format(path, id, algo)
            logger.info("------------------------------")
            logger.info("processing {}".format(file_id))
            if not os.path.exists(file_id):
                logging.warning("{} does not exist , skipping it".format(file_id))
                continue

            # Results files
            id_results = pd.DataFrame()
            for file_param in os.listdir(file_id):
                m = re.search("=(.*).result", file_param)
                logger.info("processing {}".format(file_param))
                if m:  # valid filename
                    param = m.group(1)
                    res = pd.read_csv(file_id + "/" + file_param, sep=' ', names=['R', 'C', 'Rd', 'Cd'])
                    res["parameter"] = param
                    id_results = id_results.append(res, sort=False)
                else:
                    continue
            # logger.info("id_results : \n{}".format(id_results))
            if id_results.empty:
                logging.warning("Could not find any result at {}".format(file_id))
                continue
            id_results["id"] = id
            results = results.append(id_results, sort=False)
        results["algorithm"] = algo
        data = data.append(results, sort=False)
    data.id = data.id.astype(int)
    return data


def plot_all(data, path, params):
    means = data.groupby(['algorithm', 'parameter', 'id']).mean().reset_index()
    stds = data.groupby(['algorithm', 'parameter', 'id']).std().reset_index()
    counts = data.groupby(['algorithm', 'parameter', 'id']).count().reset_index()

    means_of_means = means.groupby(['algorithm', 'parameter']).mean().reset_index()
    std_of_means = means.groupby(['algorithm', 'parameter']).std().reset_index()
    mean_of_stds = stds.groupby(['algorithm', 'parameter']).mean().reset_index()

    plot_patch(means_of_means, mean_of_stds, counts, x="Cd", y="Rd", curves="algorithm", points="parameter",
               params=params, filename=os.path.join(path, "results_disc_extra.png"))
    plot_patch(means_of_means, std_of_means, counts, x="Cd", y="Rd", curves="algorithm", points="parameter",
               params=params, filename=os.path.join(path, "results_disc_intra.png"))
    plot_patch(means_of_means, mean_of_stds, counts, x="C", y="R", curves="algorithm", points="parameter",
               params=params, filename=os.path.join(path, "results_extra.png"))
    plot_patch(means_of_means, std_of_means, counts, x="C", y="R", curves="algorithm", points="parameter",
               params=params, filename=os.path.join(path, "results_intra.png"))
    plot_lines(means, x='Cd', y='Rd', hue="algorithm", style="id", points="parameter", filename=os.path.join(path, "results_disc_ids.png"))


def plot_patch(mean, std, counts, x, y, curves, points, params, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    result = pd.concat([mean, std.add_suffix("_std"), counts.add_suffix("_count")], axis=1, sort=False)
    for group_label, data in result.groupby(curves):
        data.plot.scatter(x=x, y=y, ax=ax, c=[params[group_label][0]], zorder=2)
        for param, point in data.groupby(points):
            sqr_n = np.sqrt(point[x + "_count"].values)
            confidence_x = 1.96 * (point[x + "_std"].values / sqr_n)
            confidence_y = 1.96 * (point[y + "_std"].values / sqr_n)
            rect = patches.Rectangle((point[x].values - confidence_x, point[y].values - confidence_y),
                                     2 * confidence_x, 2 * confidence_y,
                                     linewidth=1.0,
                                     fill=True,
                                     edgecolor=(*params[group_label][0], 1),
                                     facecolor=(*params[group_label][0], 0.2), zorder=0)
            ax.add_patch(rect)
            plt.annotate("{:.2f}".format(float(param)), (point[x].values, point[y].values))
    plt.legend(handles=[patches.Patch(label=param[1], color=param[0]) for _, param in params.items()])
    if filename:
        plt.savefig(filename)
    plt.show()
    plt.close()


def plot_lines(data, x=None, y=None, filename=None, points=None, **kwargs):
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x=x, y=y, ax=ax, estimator=None, **kwargs)
    if points:
        for param, point in data.groupby(points):
            for _, row in point.iterrows():
                plt.annotate("{:.2f}".format(float(param)), (row[x], row[y]))
    if filename:
        plt.savefig(filename)
    plt.show()
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError("Usage: plot_data.py <path>")
    workspace = sys.argv[1]
    palette = itertools.cycle(sns.color_palette())
    algos = {
        "ftq": [next(palette), r"ftq($\lambda$)"],
        "bftq": [next(palette), r"bftq($\beta$)"]
    }
    data = parse_data(workspace, algos)
    print(data)
    plot_all(data, workspace, algos)
