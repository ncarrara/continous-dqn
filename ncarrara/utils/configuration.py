class Configuration(object):

    def __init__(self, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        import logging
        self.logger = logging.getLogger(__name__)
        self.dict = {}
        self.device = None
        self.plt = None

    def __getitem__(self, arg):
        self.__check__()
        return self.dict[arg]

    def __str__(self):
        import pprint
        return pprint.pformat(self.dict)

    def __check__(self):
        if not self.dict:
            raise Exception("please load the configuration file")

    def create_fresh_workspace(self):
        r = ''
        while r!='y' and r!='n':
            r = input("are you sure you want to erase workspace {} [y/n] ?".format(self.workspace))
            from ncarrara.utils.os import makedirs
            if r=='y':
                self.__check__()
                self.__clean_workspace()

            elif  r=='n':
                makedirs(self.workspace)
            else:
                print("Only [y/n]")
        return self

    def __clean_workspace(self):
        self.__check__()
        import os
        os.system("rm -rf {}".format(self.workspace))
        return self

    def load(self, path_config):
        if self.plt is None:
            import matplotlib.pyplot as plt
            self.plt = plt
        with open(path_config, 'r') as infile:
            import json
            self.dict = json.load(infile)

        self.id = self.dict["general"]["id"]
        self.workspace = self.dict["general"]["workspace"]

        self.seed = self.dict["general"]["seed"]

        import logging.config as config
        config.dictConfig(self.dict["general"]["dictConfig"])

        if self.seed is not None:
            from ncarrara.utils.math import set_seed
            set_seed(self.seed)

        import torch
        import numpy as np
        if str(torch.__version__) == "0.4.1.":
            self.logger.warning("0.4.1. is bugged regarding mse loss")
        np.set_printoptions(precision=2)
        self.logger.info("Pytorch version : {}".format(torch.__version__))
        return self

    def load_matplotlib(self, graphic_engine):
        import matplotlib
        matplotlib.use(graphic_engine)
        import matplotlib.pyplot as plt
        self.plt = plt
        return self

    def load_pytorch(self):
        from ncarrara.utils.torch import set_device
        import torch
        _device = set_device()
        self.device = torch.device("cuda:" + str(_device) if torch.cuda.is_available() else "cpu")
        self.logger.info("DEVICE : {}".format(self.device))
        return self
