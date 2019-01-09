import numpy as np
import os
import re
import matplotlib.pyplot as plt
from matplotlib import patches
import logging

logger = logging.getLogger(__name__)


def main(data_paths):
    colors = ["r", "g", "b", "purple", "grey", "black", "yellow", 'orange']
    fig, ax = plt.subplots(1, figsize=(6, 5))

    for ipath, data_path_tuple in enumerate(data_paths):
        all_xs = None
        all_ys = None
        for data_path in data_path_tuple:
            print(data_path)
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
                    # plt.annotate("{:.2f}".format(param), (ret_c, ret_r))
                    xs.append(ret_c)
                    ys.append(ret_r)
                else:
                    logger.warning("Malformed file : {}".format(file))
            if all_xs is None:
                all_xs = np.array(xs)
            else:
                all_xs = np.vstack((all_xs, np.array(xs)))
            if all_ys is None:
                all_ys = np.array(ys)
            else:
                all_ys = np.vstack((all_ys, np.array(ys)))

        print(all_xs)
        print(all_ys)
        if len(data_path_tuple)>1:
            all_xs= np.mean(all_xs, 0)
            all_ys=np.mean(all_ys, 0)
        for i,param in enumerate(params):
            plt.annotate("{:.2f}".format(param), (all_xs[i], all_ys[i]))
        plt.scatter(all_xs, all_ys, color=colors[ipath], label=r"$data_path$", zorder=2)
    plt.legend(data_paths)
    # plt.title(C["general"]["id"])
    plt.show()
    plt.savefig(data_path + "/results.png")
    plt.close()


if __name__ == "__main__":
    # main([
    #
    #     "tmp/14/bftq/results",
    #     "tmp/14/hdc/results",
    #     "tmp/14/ftq/results",
    #
    # ])

    #
    # main([
    #     ["tmp/14/hdc/results"],
    #     ["tmp/17.0/ftq/results", "tmp/17.0/ftq/results"],
    #     ["tmp/17.0/bftq/results", "tmp/17.0/bftq/results"],
    #
    # ])

    main([
        # ["tmp/camera_ready_0/hdc/results","tmp/camera_ready_1/hdc/results"],
        # ["tmp/camera_ready_0/ftq/results","tmp/camera_ready_1/ftq/results"],
         ["tmp/23.0/ftq/results"],
        ["camera_ready_0/bftq/results"],

    ])

    main([["camera_ready_1/bftq/results"]])

    #
    # main([
    #     "tmp/17.1/hdc/results",
    #     "tmp/17.1/ftq/results",
    #     "tmp/17.1/bftq/results",
    #
    # ])
    #

    # main([
    #     "tmp/results_bftq",
    #     "tmp/results_ftq",
    #
    # ])

    # for v in [7.2,11,12]:
    #
    #     main([
    #
    #         "tmp/yaya{}.2/bftq/results".format(v),
    #         "tmp/yaya{}.2/hdc/results".format(v),
    #         "tmp/yaya{}.2/ftq/results".format(v),
    #
    #         "tmp/yaya{}/bftq/results".format(v),
    #         "tmp/yaya{}/hdc/results".format(v),
    #         "tmp/yaya{}/ftq/results".format(v),
    #
    #
    #     ])
