import os
from configuration import C


def main():
    os.system("rm -rf " + C.workspace)
