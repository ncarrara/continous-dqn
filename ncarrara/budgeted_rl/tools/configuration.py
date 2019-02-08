from ncarrara.utils.configuration import Configuration


class ConfigurationBFTQ(Configuration):
    import logging
    logger = logging.getLogger(__name__)

    def create_fresh_workspace(self,force=False):
        super(ConfigurationBFTQ, self).create_fresh_workspace(force)
        # from ncarrara.utils.os import makedirs
        # makedirs(self.path_bftq)
        # makedirs(self.path_bftq_results)
        # makedirs(self.path_ftq)
        # makedirs(self.path_ftq_results)
        # makedirs(self.path_hdc)
        # makedirs(self.path_hdc_results)
        # makedirs(self.path_dqn)
        # makedirs(self.path_dqn_results)
        # makedirs(self.path_bdqn)
        # makedirs(self.path_bdqn_results)
        # makedirs(self.path_learn_bftq_egreedy)
        # makedirs(self.path_learn_ftq_egreedy)
        return self

    def load(self, config):
        super(ConfigurationBFTQ, self).load(config)
        self.path_hdc_results = self.workspace + "/hdc/results"
        self.path_hdc = self.workspace + "/hdc"
        self.path_ftq_results = self.workspace + "/ftq/results"
        self.path_ftq = self.workspace + "/ftq"
        self.path_bftq_results = self.workspace + "/bftq/results"
        self.path_bftq = self.workspace + "/bftq"
        self.path_dqn_results = self.workspace + "/dqn/results"
        self.path_dqn = self.workspace + "/dqn"
        self.path_bdqn_results = self.workspace + "/bdqn/results"
        self.path_bdqn = self.workspace + "/bdqn"
        self.path_learn_bftq_egreedy = self.path_bftq + "/learn_bftq_egreedy"
        self.path_learn_ftq_egreedy = self.path_ftq + "/learn_ftq_egreedy"
        return self


C = ConfigurationBFTQ() #.load_pytorch()  # BFTQ_PYDIAL().load_pytorch()
