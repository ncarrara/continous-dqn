# -*- coding: utf-8 -*-
import numpy as np
import torch
import copy
import torch.nn.functional as F

from ncarrara.continuous_dqn.tools.configuration import C
from ncarrara.utils.torch import optimizer_factory, BaseModule
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.utils_rl.transition.transition import TransitionGym
import logging

logger = logging.getLogger(__name__)


class NetDQN(BaseModule):
    def __init__(self, n_in, n_out, intra_layers, activation_type="RELU", reset_type="XAVIER", normalize=None):
        super(NetDQN, self).__init__(activation_type, reset_type, normalize)
        all_layers = [n_in] + intra_layers + [n_out]
        self.layers = []
        for i in range(0, len(all_layers) - 2):
            module = torch.nn.Linear(all_layers[i], all_layers[i + 1])
            self.layers.append(module)
            self.add_module("h_" + str(i), module)
        self.predict = torch.nn.Linear(all_layers[-2], all_layers[-1])

    def forward(self, x):
        if self.normalize:
            x = (x.float() - self.mean.float()) / self.std.float()
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.predict(x)
        return x.view(x.size(0), -1)


class DQN:
    ALL_BATCH = "ALL_BATCH"
    ADAPTATIVE = "ADAPTATIVE"

    def __init__(self,
                 device,
                 policy_network,
                 gamma=0.999,
                 batch_size_experience_replay=128,
                 target_update=10,
                 optimizer=None,
                 loss_function=None,
                 workspace=None,
                 lr=None,
                 weight_decay=None,
                 transfer_module = None,
                 **kwargs):
        self.tranfer_module=transfer_module
        self.device= device
        self.size_mini_batch = batch_size_experience_replay
        self.GAMMA = gamma
        self.TARGET_UPDATE = target_update
        self.workspace = workspace
        self.policy_net = policy_network.to(self.device)
        self.target_net = copy.deepcopy(policy_network)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optimizer
        self.memory = Memory()
        self.i_episode = 0
        self.no_need_for_transfer_anymore = False
        self.n_actions = self.policy_net.predict.out_features
        # self.transfer_experience_replay = None

        if loss_function == "l2":
            self.loss_function = F.mse_loss
        elif loss_function == "l1":
            self.loss_function = F.l1_loss
        else:
            raise Exception("Unknown loss function : {}".format(loss_function))
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer
        self.optimizer = None
        self.reset()


    def update_transfer_experience_replay(self, er):
        self.transfer_experience_replay = er

    def reset(self, reset_weight=True):
        self.memory.reset()
        if reset_weight:
            self.policy_net.reset()
        self.optimizer = optimizer_factory(self.optimizer_type,
                                           self.policy_net.parameters(),
                                           self.lr,
                                           self.weight_decay)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.i_episode = 0
        self.transfer_experience_replay = None
        self.no_need_for_transfer_anymore = False

    def _optimize_model(self):

        transitions = self.memory.sample(len(self.memory) if self.size_mini_batch > len(self.memory) else self.size_mini_batch)

        if self.tranfer_module is not None:
            # reeavalute errors
            # TODO

            # transfert samples
            if self.tranfer_module.get_experience_replay_source() is not None:
                size_transfer = self.size_mini_batch - len(transitions)
                if size_transfer > 0 and not self.no_need_for_transfer_anymore:
                    transfer_transitions = self.tranfer_module.get_experience_replay_source().sample(size_transfer)
                    transitions = transitions + transfer_transitions
                self.no_need_for_transfer_anymore = size_transfer <= 0 and self.transfer_experience_replay is not None

            # transfer Q
            # TODO

        batch = TransitionGym(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)),
                                      device=self.device,
                                      dtype=torch.uint8)
        non_final_next_states= [s for s in batch.s_ if s is not None]
        # non_final_next_states = torch.cat(non_final_next_states)
        self._state_batch = torch.cat(batch.s)
        self._action_batch = torch.cat(batch.a)
        reward_batch = torch.cat(batch.r_)
        state_action_values = self.policy_net(self._state_batch).gather(1, self._action_batch)
        next_state_values = torch.zeros(len(transitions), device=self.device)
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
            action_mask = torch.tensor([action_mask], device=self.device, dtype=torch.float)
            s = torch.tensor([[state]], device=self.device, dtype=torch.float)
            a = self.policy_net(s).sub(action_mask).max(1)[1].view(1, 1).item()
            return a

    def update(self, *sample):
        state, action, reward, next_state, done, info = sample
        state = torch.tensor([[state]], device=self.device, dtype=torch.float)
        if not done:
            next_state = torch.tensor([[next_state]], device=self.device, dtype=torch.float)
        else:
            next_state = None
        action = torch.tensor([[action]], device=self.device, dtype=torch.long)
        reward = torch.tensor([float(reward)], device=self.device, dtype=torch.float)
        t = state, action, reward, next_state, done, info
        self.memory.push(*t)
        if self.tranfer_module is not None:
            self.tranfer_module.push(*t)
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
