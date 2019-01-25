# -*- coding: utf-8 -*-
import numpy as np
import torch
import copy
import torch.nn.functional as F

from ncarrara.continuous_dqn.tools.configuration import C
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.utils_rl.transition.transition import TransitionGym
import logging

logger = logging.getLogger(__name__)


class NetDQN(torch.nn.Module):
    DONT_NORMALIZE_YET = None

    def _init_weights(self, m):
        if hasattr(m, 'weight'):

            if self.reset_type == "XAVIER":
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif self.reset_type == "ZEROS":
                torch.nn.init.constant_(m.weight.data, 0.)
            else:
                raise ("fuck off mate !")
                # torch.nn.init.constant_(m.weight.data, 0.)

    def __init__(self, n_in, n_out, intra_layers, activation_type="RELU", normalize=None, reset_type="XAVIER"):
        super(NetDQN, self).__init__()
        self.reset_type = reset_type
        if activation_type == "RELU":
            activation_type = F.relu
        else:
            raise Exception("Unknow activation_type : {}".format(F.relu))
        all_layers = [n_in] + intra_layers + [n_out]
        # self.logger.info(all_layers)
        self.activation = activation_type
        self.normalize = normalize
        self.layers = []
        for i in range(0, len(all_layers) - 2):
            module = torch.nn.Linear(all_layers[i], all_layers[i + 1])
            self.layers.append(module)
            self.add_module("h_" + str(i), module)

        self.predict = torch.nn.Linear(all_layers[-2], all_layers[-1])

    def set_normalization_params(self, mean, std):
        if self.normalize:
            std[std == 0.] = 1.  # on s'en moque, on divisera 0 par 1.
        self.std = std
        self.mean = mean

    def forward(self, x):
        # self.logger.info "x : ",x
        if self.normalize:  # hasattr(self, "normalize"):
            x = (x.float() - self.mean.float()) / self.std.float()
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.predict(x)  # linear output
        return x.view(x.size(0), -1)

    def reset(self):
        self.apply(self._init_weights)


class DQN:
    ALL_BATCH = "ALL_BATCH"
    ADAPTATIVE = "ADAPTATIVE"

    def __init__(self,
                 policy_network,
                 gamma=0.999,
                 batch_size_experience_replay=128,
                 target_update=10,
                 optimizer=None,
                 loss_function=None,
                 workspace=None,
                 lr=None,
                 weight_decay=None,
                 **kwargs):
        self.BATCH_SIZE_EXPERIENCE_REPLAY = batch_size_experience_replay
        self.GAMMA = gamma
        self.TARGET_UPDATE = target_update
        self.workspace = workspace
        self.policy_net = policy_network.to(C.device)
        self.target_net = copy.deepcopy(policy_network)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optimizer
        self.memory = Memory()
        self.i_episode = 0
        self.no_need_for_transfer_anymore = False
        self.n_actions = self.policy_net.predict.out_features
        self.transfer_experience_replay = None

        if loss_function == "l2":
            self.loss_function = F.mse_loss
        elif loss_function == "l1":
            self.loss_function = F.l1_loss
        else:
            raise Exception("Unknown loss function : {}".format(loss_function))

        if optimizer == "ADAM":
            self.optimizer = torch.optim.Adam(params=self.policy_net.parameters(),
                                              lr=lr, weight_decay=weight_decay)
        elif optimizer == "RMS_PROP":
            self.optimizer = torch.optim.RMSprop(params=self.policy_net.parameters(),
                                                 weight_decay=weight_decay)
        else:
            raise Exception("Unknown optimizer : {}".format(optimizer))

    def update_transfer_experience_replay(self, er):
        self.transfer_experience_replay = er

    def reset(self, reset_weight=True):
        self.memory.reset()
        if reset_weight:
            self.policy_net.reset()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.i_episode = 0
        self.transfer_experience_replay = None
        self.no_need_for_transfer_anymore = False

    def _optimize_model(self):
        # if self.BATCH_SIZE_EXPERIENCE_REPLAY == self.ADAPTATIVE:
        #     size_batch = len(self.memory) / 10
        # elif self.BATCH_SIZE_EXPERIENCE_REPLAY == self.ALL_BATCH:
        #     size_batch = len(self.memory)
        # else:
        size_batch = self.BATCH_SIZE_EXPERIENCE_REPLAY

        transitions = self.memory.sample(len(self.memory) if size_batch > len(self.memory) else size_batch)

        size_transfer = size_batch - len(transitions)
        if size_transfer > 0 and self.transfer_experience_replay is not None and not self.no_need_for_transfer_anymore:
            transfer_transitions = self.transfer_experience_replay.sample(size_transfer)
            transitions = transitions + transfer_transitions

        self.no_need_for_transfer_anymore = size_transfer <= 0 and self.transfer_experience_replay is not None

        batch = TransitionGym(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)),
                                      device=C.device,
                                      dtype=torch.uint8)
        non_final_next_states= [s for s in batch.s_ if s is not None]
        # non_final_next_states = torch.cat(non_final_next_states)
        self._state_batch = torch.cat(batch.s)
        self._action_batch = torch.cat(batch.a)
        reward_batch = torch.cat(batch.r_)
        state_action_values = self.policy_net(self._state_batch).gather(1, self._action_batch)
        next_state_values = torch.zeros(len(transitions), device=C.device)
        if non_final_next_states:
            next_state_values[non_final_mask] = self.target_net(torch.cat(non_final_next_states)).max(1)[0].detach()
        else:
            logger.warning("Pas d'état non terminaux")
        self.expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        loss = self.loss_function(state_action_values, self.expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def pi(self, state, action_mask):

        with torch.no_grad():
            if not type(action_mask) == type(np.zeros(1)):
                action_mask = np.asarray(action_mask)
            action_mask[action_mask == 1.] = np.infty
            action_mask = torch.tensor([action_mask], device=C.device, dtype=torch.float)
            s = torch.tensor([[state]], device=C.device, dtype=torch.float)
            a = self.policy_net(s).sub(action_mask).max(1)[1].view(1, 1).item()
            return a

    def update(self, *sample):
        state, action, reward, next_state, done, info = sample
        state = torch.tensor([[state]], device=C.device, dtype=torch.float)
        if not done:
            next_state = torch.tensor([[next_state]], device=C.device, dtype=torch.float)
        else:
            next_state = None
        action = torch.tensor([[action]], device=C.device, dtype=torch.long)
        reward = torch.tensor([float(reward)], device=C.device, dtype=torch.float)
        self.memory.push(state, action, reward, next_state, done, info)
        self._optimize_model()
        if next_state is None:
            self.i_episode += 1
            if self.i_episode % self.TARGET_UPDATE == 0:
                logger.info("[update][i_episode={}] copying weights to target network".format(self.i_episode))
                self.target_net.load_state_dict(self.policy_net.state_dict())

                # if self.i_episode % 100 == 0:
                #     with torch.no_grad():
                #         print("Creating histograms ...")
                #         QQ = self.policy_net(self._state_batch)
                #         utils_dqn.create_Q_histograms(
                #             title="Q(s)_pred_target_e={}".format(self.i_episode),
                #             values=[self.expected_state_action_values.cpu().numpy(),
                #                     QQ.gather(1, self._action_batch).cpu().numpy().flat],
                #             path=self.workspace,
                #             labels=["target", "prediction"])
                #
                #         utils_dqn.create_Q_histograms_for_actions(
                #             title="actions_Q(s)_pred_target_e={}".format(self.i_episode),
                #             QQ=QQ.cpu().numpy(),
                #             path=self.workspace,
                #             labels=[str(act) for act in range(self.n_actions)],
                #             mask_action=np.zeros(self.n_actions))
