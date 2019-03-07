import os

from ncarrara.continuous_dqn.dqn.utils_dqn import run_dqn
from ncarrara.utils.os import makedirs
from ncarrara.utils_rl.environments.envs_factory import generate_envs
from ncarrara.continuous_dqn.tools import utils

import numpy as np
import logging

from ncarrara.continuous_dqn.tools.features import build_feature_autoencoder, build_feature_dqn
from ncarrara.utils.math_utils import epsilon_decay
import re
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns


def main(workspace=None):
    rez = pd.DataFrame()
    for x in os.listdir(workspace):
        if re.match("[0-9]{1,}",x):
            df = pd.read_pickle(workspace / x / "data.pd")
            df = pd.concat([df], keys=[x], names=['seed'])
            rez = rez.append(df)

    xx = df.mean(level=["config", "is_greedy"])
    xx = xx.iloc[xx.index.get_level_values('is_greedy') == True]
    xx = xx.reset_index('is_greedy', drop=True).T
    xx.plot()
    plt.savefig(workspace / "results_greedy")
    xx = df.mean(level=["config", "is_greedy"])
    xx = xx.iloc[xx.index.get_level_values('is_greedy') == False]
    xx = xx.reset_index('is_greedy', drop=True).T
    xx.plot()
    plt.savefig(workspace / "results")

from pathlib import Path

p = Path("tmp")
main(p / "cartpole" / "hard")
main(p / "cartpole" / "easy")
