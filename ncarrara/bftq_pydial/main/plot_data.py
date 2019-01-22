import numpy as np
import os
import re
import matplotlib.pyplot as plt
from matplotlib import patches
import logging

logger = logging.getLogger(__name__)


def main(data_paths):
    # colors = ["r", "g", "b", "purple", "grey", "black", "yellow", 'orange']
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
                                             # edgecolor=colors[ipath],
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
        if len(data_path_tuple) > 1:
            all_xs = np.mean(all_xs, 0)
            all_ys = np.mean(all_ys, 0)
        for i, param in enumerate(params):
            plt.annotate("{:.2f}".format(param), (all_xs[i], all_ys[i]))
        plt.scatter(all_xs, all_ys, label=r"$data_path$", zorder=2)  # ,color=colors[ipath],)
    # plt.legend(data_paths,loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.title(C["general"]["id"])
    plt.show()
    plt.savefig(data_path + "/results.png")
    plt.close()


if __name__ == "__main__":

    todo = [[],[],[]]
    # folder="camera_ready_6"
    # values = [0,1,2,3,4,7]

    folder = "camera_ready_7"
    values = [0, 1]
    for i in values:
        main([
            ["tmp/{}/{}/ftq/results".format(folder,i)],
            ["tmp/{}/{}/bftq/results".format(folder,i)],
            ["tmp/{}/{}/hdc/results".format(folder,i)]],
        )

        todo[0].append("tmp/{}/{}/ftq/results".format(folder,i))
        todo[1].append("tmp/{}/{}/bftq/results".format(folder,i))
        todo[2].append("tmp/{}/{}/hdc/results".format(folder,i))
    main(todo)

