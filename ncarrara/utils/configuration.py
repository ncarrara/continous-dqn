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

    def create_fresh_workspace(self,force=False):
        r = ''
        while r!='y' and r!='n':
            if force:
                r='y'
            else:
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

    def load(self, config):
        if self.plt is None:
            import matplotlib.pyplot as plt
            self.plt = plt
        if type(config) == type(""):
            self.logger.info("reading config file at {}".format(config))
            with open(config, 'r') as infile:
                import json
                self.dict = json.load(infile)
        elif type(config) == type({}):
            self.dict = config
        else:
            raise TypeError("Wrong type for configuration, must be a path or a dict")

        self.id = self.dict["general"]["id"]
        self.workspace = self.dict["general"]["workspace"]



        import logging.config as config
        config.dictConfig(self.dict["general"]["dictConfig"])

        # if seed is not None:
        #     self.logger.warning("overriding seed")
        self.seed = self.dict["general"]["seed"]
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
        from ncarrara.utils.torch import get_the_device_with_most_available_memory
        _device = get_the_device_with_most_available_memory()
        self.device = _device
        self.logger.info("DEVICE : {}".format(self.device))
        return self
