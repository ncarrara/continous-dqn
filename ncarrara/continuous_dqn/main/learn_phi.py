from ncarrara.budgeted_rl.tools.utils_run import datas_to_transitions
from ncarrara.continuous_dqn.tools import utils as utils, features

import os
import logging

from ncarrara.continuous_dqn.tools.features import build_feature_autoencoder
from ncarrara.utils.os import makedirs
from ncarrara.utils.torch_utils import loss_fonction_factory, optimizer_factory
from ncarrara.utils_rl.algorithms.pytorch_fittedq import PytorchFittedQ, NetFTQ

logger = logging.getLogger(__name__)


def main(loss_function_str, optimizer_str, weight_decay, learning_rate, normalize,
         autoencoder_size, n_epochs, feature_autoencoder_info, workspace, device, type_ae="AEA", N_actions=None,
         writer=None):
    import torch
    loss_function = loss_fonction_factory(loss_function_str)
    makedirs(workspace)
    feature = build_feature_autoencoder(feature_autoencoder_info)

    all_transitions = utils.load_memories(workspace / "samples", as_json=False)
    all_transitions_ftq=[]


    for transitions in all_transitions:
        transitions_ftq = datas_to_transitions(transitions, env, feature, lambda_, normalize_reward)
        all_transitions_ftq.extend(transitions_ftq)

    net = NetFTQ(n_in=len(feature(e.reset(), e)), n_out=e.action_space.n, **ftq_net_params)
    ftq = PytorchFittedQ(
        device=device,
        policy_network=net,
        action_str=None if not hasattr(e, "action_str") else e.action_str,
        test_policy=None,
        gamma=gamma,
        **ftq_params
    )

    ftq.reset(True)
    ftq.fit(all_transitions_ftq)

    phi = net[:-2]

    path_phi = workspace / "phi.pt"
    logger.info("saving autoencoder at {}".format(path_phi))
    torch.save(phi, path_phi)
