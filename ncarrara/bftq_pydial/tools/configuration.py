from ncarrara.utils.configuration import Configuration


class ConfigurationBFTQ_PYDIAL(Configuration):
    import logging
    logger = logging.getLogger(__name__)

    def create_fresh_workspace(self,force):
        super(ConfigurationBFTQ_PYDIAL, self).create_fresh_workspace(force)
        from ncarrara.utils.os import makedirs
        makedirs(self.path_bftq)
        makedirs(self.path_bftq_results)
        makedirs(self.path_ftq)
        makedirs(self.path_ftq_results)
        makedirs(self.path_hdc)
        makedirs(self.path_hdc_results)

    def load(self, config):
        super(ConfigurationBFTQ_PYDIAL, self).load(config)
        self.path_hdc_results = self.workspace + "/hdc/results"
        self.path_hdc = self.workspace + "/hdc"
        self.path_ftq_results = self.workspace + "/ftq/results"
        self.path_ftq = self.workspace + "/ftq"
        self.path_bftq_results = self.workspace + "/bftq/results"
        self.path_bftq = self.workspace + "/bftq"
        return self


C = ConfigurationBFTQ_PYDIAL().load_pytorch()  # BFTQ_PYDIAL().load_pytorch()
