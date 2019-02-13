from ncarrara.utils.configuration import Configuration
import os

class ConfigurationBFTQ(Configuration):
    import logging
    logger = logging.getLogger(__name__)

    def create_fresh_workspace(self, force=False):
        super(ConfigurationBFTQ, self).create_fresh_workspace(force)
        return self

    def load(self, config):
        super(ConfigurationBFTQ, self).load(config)
        self.update_paths()
        return self

    def update_paths(self):
        self.id_bftq_egreedy= "bftq_egreedy"
        self.id_ftq_egreedy= "ftq_egreedy"
        self.id_ftq_duplicate= "ftq_duplicate"
        self.result_folder = "results"

        self.path_bftq_egreedy = os.path.join(self.workspace, self.id_bftq_egreedy)
        self.path_bftq_egreedy_results = os.path.join(self.workspace, self.id_bftq_egreedy, self.result_folder)
        self.path_ftq_egreedy = os.path.join(self.workspace, self.id_ftq_egreedy)
        self.path_ftq_egreedy_results = os.path.join(self.workspace, self.id_ftq_egreedy, self.result_folder)
        self.path_ftq_duplicate = os.path.join(self.workspace, self.id_ftq_duplicate)
        self.path_ftq_duplicate_results = os.path.join(self.workspace, self.id_ftq_duplicate, self.result_folder)



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


C = ConfigurationBFTQ()
