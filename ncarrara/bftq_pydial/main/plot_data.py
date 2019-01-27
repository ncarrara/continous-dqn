import numpy as np
import os
import re
import matplotlib.pyplot as plt
from matplotlib import patches
import logging
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)


def main(data_dict):
    # colors = ["r", "g", "b", "purple", "grey", "black", "yellow", 'orange']
    fig, ax = plt.subplots(1, figsize=(6, 5))

    datas = {}

    for algo_str, v in data_dict.items():
        file_seeds, _, params_to_search,_ = v
        datas_algo = []
        for file_seed in file_seeds:
            datas_seed = []
            logger.info("processing {}".format(file_seed))
            if not os.path.exists(file_seed):
                logging.warning("{} do not exists, skipping it".format(file_seed))
            else:
                files_lambda = os.listdir(file_seed)
                if not files_lambda:
                    logging.warning("{} exists, but no data, skipping it".format(file_seed))
                else:
                    for file_lambda in files_lambda:
                        m = re.search("=(.*).results", file_lambda)
                        if m:
                            param = float(m.group(1))
                            results = np.loadtxt(file_seed + "/" + file_lambda, np.float32)
                            datas_seed.append((results, param))
                        else:
                            logger.warning("Malformed file : {}".format(file_lambda))
                    if not datas_seed:
                        logging.warning("malformed results at {}".format(file_seed))
                    else:
                        datas_seed.sort(key=lambda tup: tup[1])
                        # print(datas_seed)
                        datas_seed, params = zip(*datas_seed)
                        if params == params_to_search:
                            datas_algo.append(datas_seed)
                        else:
                            logging.warning(("malformed params, {} != {}".format(params, params_to_search)))
        datas[algo_str] = (np.asarray(datas_algo), params_to_search)

    patchList = []

    for algo_str, v in datas.items():
        data, params = v
        patch = mpatches.Patch(color=data_dict[algo_str][1], label=r"$"+algo_str+"("+data_dict[algo_str][3]+")$")
        patchList.append(patch)
        N_seed = data.shape[1]
        means_run = np.mean(data, 2)
        means_seed = np.mean(means_run, 0)
        mean_std_seed = np.std(means_run, 0)
        for iparam, param in enumerate(params):
            mean = means_seed[iparam]
            std = mean_std_seed[iparam]
            x, y = mean[3], mean[2]
            std_x, std_y = std[3], std[2]
            plt.scatter(x, y,  zorder=2, c=data_dict[algo_str][1])  # ,color=colors[ipath],)
            confidence_y = 1.96 * (std_y / np.sqrt(N_seed))
            confidence_x = 1.96 * (std_x / np.sqrt(N_seed))
            rect = patches.Rectangle((x - confidence_x, y - confidence_y),
                                     2 * confidence_x,
                                     2 * confidence_y,
                                     linewidth=1.0,
                                     fill=True,
                                     edgecolor=data_dict[algo_str][1] + (1,),
                                     facecolor=data_dict[algo_str][1] + (0.2,), zorder=0)
            ax.add_patch(rect)
            plt.annotate("{:.2f}".format(param), (x, y))
    plt.legend(handles=patchList)
    plt.grid()
    plt.show()
    plt.close()


if __name__ == "__main__":

    data = {
        "ftq": ([],
                (1, 0, 0),
                (0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0, 250.0, 500.0),
                "\lambda"),
        "bftq": ([],
                 (0, 1, 0),
                 (0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32,
                  0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.6, 0.64, 0.68,
                  0.72, 0.76, 0.8, 0.84, 0.88, 0.92, 0.96, 1.0),
                 "\\beta"
                 ),
        # "hdc": ([],
        #         (0, 0, 1),
        #         (0.0, 0.1111111111111111, 0.2222222222222222, 0.3333333333333333, 0.4444444444444444,
        #          0.5555555555555556, 0.6666666666666666, 0.7777777777777777, 0.8888888888888888, 1.0),
        #         "safeness"
        #         )
    }
    folder = "camera_ready_6.2"
    values = range(0,18)

    # folder = "camera_ready_7"
    # values = [0, 1]
    for i in values:
        # main([
        #     ["tmp/{}/{}/ftq/results".format(folder,i)],
        #     ["tmp/{}/{}/bftq/results".format(folder,i)],
        #     ["tmp/{}/{}/hdc/results".format(folder,i)]],
        # )
        data["ftq"][0].append("tmp/{}/{}/ftq/results".format(folder, i))
        data["bftq"][0].append("tmp/{}/{}/bftq/results".format(folder, i))
        # data["hdc"][0].append("tmp/{}/{}/hdc/results".format(folder, i))
    main(data)
