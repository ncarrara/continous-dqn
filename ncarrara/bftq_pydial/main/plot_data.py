import matplotlib

matplotlib.use("Agg")
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import patches
import logging
import matplotlib.patches as mpatches
import re

logger = logging.getLogger(__name__)


def main(path, params_algos):
    with open(path + "/" + "params") as f:
        lines = f.readlines()
        match = re.search("^id=([0-9]+) .*$", lines[-1])
        nb_ids = int(match.group(1)) + 1

    fig, ax = plt.subplots(1, figsize=(6, 5))
    plt.grid()

    datas = []

    for ialgo, paramalgo in enumerate(params_algos):
        algo_str, _, _ = paramalgo
        params_to_search = []
        for path_param in os.listdir("{}/{}/{}/results".format(path, 0, algo_str)):
            match = re.search("=(.*).results", path_param)
            params_to_search.append(match.group(1))
        params_to_search.sort()
        paramalgo.append(params_to_search)
    # print(params_algos)
    # exit()
    skipthoseids = np.zeros(nb_ids)
    for algo_str, _, _, params_to_search in params_algos:

        datas_algo = [None] * nb_ids
        for id in range(nb_ids):
            file_id = "{}/{}/{}/results".format(path, id, algo_str)
            datas_id = []
            logger.info("processing {}".format(file_id))
            if not os.path.exists(file_id):
                logging.warning("{} do not exists, skipping it".format(file_id))
            else:
                files_params = os.listdir(file_id)
                if not files_params:
                    logging.warning("{} exists, but no data, skipping it".format(file_id))
                    skipthoseids[id] = True
                else:
                    for file_param in files_params:
                        m = re.search("=(.*).results", file_param)
                        if m:
                            param = m.group(1)
                            results = np.loadtxt(file_id + "/" + file_param, np.float32)
                            datas_id.append((results, param))
                        else:
                            logger.warning("Malformed file : {}".format(file_param))
                            skipthoseids[id] = True
                    if not datas_id:
                        logging.warning("malformed results at {}".format(file_id))
                        skipthoseids[id] = True
                    else:
                        datas_id.sort(key=lambda tup: tup[1])
                        # print(datas_seed)
                        data_algo, params = zip(*datas_id)
                        params = list(params)
                        if params == params_to_search:
                            datas_algo[id] = data_algo
                        else:
                            logging.warning(("malformed params, {} != {}".format(params, params_to_search)))
                            skipthoseids[id] = True
                        # exit()
        datas.append([algo_str, datas_algo, params_to_search])

    print("malformed ids :", skipthoseids)

    for ientry, entry in enumerate(datas):
        algo_str, d, params_to_search = entry
        datas_algo = []
        for idd, dd in enumerate(d):
            if not skipthoseids[idd]:
                datas_algo.append(dd)
        entry[1] = np.asarray(datas_algo)

    # exit()

    patchList = []

    for ialgo, dataalgo in enumerate(datas):
        algo_str, data, params_to_search = dataalgo
        print(data.shape)
        patch = mpatches.Patch(color=params_algos[ialgo][1],
                               label=r"$" + algo_str + "(" + params_algos[ialgo][2] + ")$")
        patchList.append(patch)
        N_seed = data.shape[0]
        print("N_seed : {}".format(N_seed))
        means_intra_seed = np.mean(data, 2)
        stds_intra_seed = np.std(data, 2)

        means_ids = np.mean(means_intra_seed, 0)
        std_extra_seed = np.std(means_intra_seed, 0)
        std_intra_seed = np.mean(stds_intra_seed, 0)


        # mean_std_ids = np.std(means_trajectories, 0)
        for iparam, param in enumerate(params_to_search):
            mean = means_ids[iparam]
            std = std_extra_seed[iparam]
            x, y = mean[3], mean[2]
            std_x, std_y = std[3], std[2]
            plt.scatter(x, y, zorder=2, c=params_algos[ialgo][1])  # ,color=colors[ipath],)
            confidence_y = 1.96 * (std_y / np.sqrt(N_seed))
            confidence_x = 1.96 * (std_x / np.sqrt(N_seed))
            rect = patches.Rectangle((x - confidence_x, y - confidence_y),
                                     2 * confidence_x,
                                     2 * confidence_y,
                                     linewidth=1.0,
                                     fill=True,
                                     edgecolor=params_algos[ialgo][1] + [1],
                                     facecolor=params_algos[ialgo][1] + [0.2], zorder=0)
            ax.add_patch(rect)

            plt.annotate("{:.2f}".format(float(param)), (x, y))
    plt.legend(handles=patchList)
    plt.show()
    plt.savefig(path + "/" + "results_extra.png")
    plt.close()
    fig, ax = plt.subplots(1, figsize=(6, 5))
    plt.grid()
    patchList = []

    for ialgo, dataalgo in enumerate(datas):
        algo_str, data, params_to_search = dataalgo
        print(data.shape)
        patch = mpatches.Patch(color=params_algos[ialgo][1],
                               label=r"$" + algo_str + "(" + params_algos[ialgo][2] + ")$")
        patchList.append(patch)
        N_seed = data.shape[2]
        print("N_traj : {}".format(N_seed))
        means_intra_seed = np.mean(data, 2)
        stds_intra_seed = np.std(data, 2)

        means_ids = np.mean(means_intra_seed, 0)
        std_extra_seed = np.std(means_intra_seed, 0)
        std_intra_seed = np.mean(stds_intra_seed, 0)

        # mean_std_ids = np.std(means_trajectories, 0)
        for iparam, param in enumerate(params_to_search):
            mean = means_ids[iparam]
            std = std_intra_seed[iparam]
            x, y = mean[3], mean[2]
            std_x, std_y = std[3], std[2]
            plt.scatter(x, y, zorder=2, c=params_algos[ialgo][1])  # ,color=colors[ipath],)
            confidence_y = 1.96 * (std_y / np.sqrt(N_seed))
            confidence_x = 1.96 * (std_x / np.sqrt(N_seed))
            rect = patches.Rectangle((x - confidence_x, y - confidence_y),
                                     2 * confidence_x,
                                     2 * confidence_y,
                                     linestyle="--",
                                     linewidth=1.0,
                                     fill=True,
                                     edgecolor=params_algos[ialgo][1] + [1],
                                     facecolor=params_algos[ialgo][1] + [0.2], zorder=0)
            ax.add_patch(rect)

            plt.annotate("{:.2f}".format(float(param)), (x, y))
    plt.legend(handles=patchList)
    plt.show()
    plt.savefig(path + "/" + "results_intra.png")


    plt.close()


if __name__ == "__main__":
    path = "tmp/camera_ready_6.2"
    params = (
        ["ftq", [1, 0, 0], "\lambda"],
        ["bftq", [0, 1, 0], "\\beta"],
        ["hdc", [0, 0, 1], "safeness"]
    )
    main(path, params)
