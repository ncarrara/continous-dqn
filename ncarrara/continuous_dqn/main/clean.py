import os

from ncarrara.continuous_dqn.tools.configuration import C


def main():
    os.system("rm -rf " + C.workspace)
