import numpy as np
import os
import re
import matplotlib.pyplot as plt
from matplotlib import patches
import logging


logger = logging.getLogger(__name__)


def main(data_paths):
    colors=["r","g","b","purple","grey","black","yellow",'orange']
    fig, ax = plt.subplots(1, figsize=(6, 5))
    for ipath,data_path in enumerate(data_paths):
        logger.info("processing {}".format(data_path))
        files = os.listdir(data_path)
        params = []
        xs = []
        ys = []
        for file in files:
            m = re.search("=(.*).results", file)
            if m:
                param = float(m.group(1))
                params.append(param)
                results = np.loadtxt(data_path + "/" + file, np.float32)
                N = len(results)
                rew_r, rew_c, ret_r, ret_c = np.mean(results, axis=0)
                std_rew_r, std_rew_c, std_ret_r, std_ret_c = np.std(results, axis=0)
                confidence_r = 1.96 * (std_ret_r / np.sqrt(N))
                confidence_c = 1.96 * (std_ret_c / np.sqrt(N))

                rect = patches.Rectangle((ret_c - confidence_c, ret_r - confidence_r),
                                         2 * confidence_c,
                                         2 * confidence_r,
                                         linewidth=1.0,
                                         edgecolor=colors[ipath],
                                         fill=True,
                                         facecolor=(1, 0, 0, 0.0), zorder=0)
                # ax.add_patch(rect)
                plt.annotate("{:.2f}".format(param), (ret_c, ret_r))
                xs.append(ret_c)
                ys.append(ret_r)
            else:
                logger.warning("Malformed file : {}".format(file))

        plt.scatter(xs, ys, color=colors[ipath], label=r"$data_path$", zorder=2)
    plt.legend(data_paths)
    # plt.title(C["general"]["id"])
    plt.show()
    plt.savefig(data_path + "results.png")
    plt.close()
