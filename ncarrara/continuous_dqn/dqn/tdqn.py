# -*- coding: utf-8 -*-
import numpy as np
import torch
import copy
import torch.nn.functional as F

from ncarrara.utils.os import makedirs
from ncarrara.utils.torch_utils import BaseModule, optimizer_factory, loss_fonction_factory
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.utils_rl.transition.transition import TransitionGym
import logging

logger = logging.getLogger(__name__)


class TranferNetwork(BaseModule):
    def __init__(self, Q, reset_type="XAVIER"):
        super(TranferNetwork, self).__init__("SIGMOID", reset_type, normalize=False)
        self.Q = Q

        self.n_actions = Q.predict.out_features
        self.predict_Q = torch.nn.Linear(self.n_actions + 1, self.n_actions)
        self.predict_Q_s = torch.nn.Linear(self.n_actions + 1, self.n_actions)
        self.actions = []
        for i in range(self.n_actions):
            module = torch.nn.Linear(2, 1)
            self.actions.append(module)
            self.add_module("a_" + str(i), module)
        # print(self)
        # exit()

    def set_Q_source(self, Q_source, ae_score):
        self.Q_source = Q_source
        self.ae_score = ae_score

    def reset(self):
        self.Q_source = None
        self.ae_score = np.nan
        super(TranferNetwork, self).reset()

    def forward(self, s):
        # print("s.shape",s.shape)
        Q = self.Q(s)
        Q_s = self.Q_source(s)
        # print(Q.shape)
        # print(self.ae_score.unsqueeze(0).shape)
        # print(Q_source.shape)
        ae_score = self.ae_score.repeat(len(s)).unsqueeze(1)
        # print("ae_score.shape",ae_score.shape)
        Q_and_ae = torch.cat((Q, ae_score), 1)
        # print("Q",Q)
        # print("ae_score",ae_score)
        # print("Q_and_ae",Q_and_ae)
        out_Q_and_ae = self.activation(self.predict_Q(Q_and_ae))
        # print("out_Q_and_ae",out_Q_and_ae)
        Q_s_and_ae = torch.cat((Q_s, ae_score), 1)
        out_Q_s_and_ae = self.activation(self.predict_Q_s(Q_s_and_ae))
        rez = []
        for i in range(self.n_actions):
            # print("out_Q_and_ae.shape",out_Q_and_ae.shape)
            # print(out_Q_source_ae)
            v_act_s = out_Q_s_and_ae[:, i].unsqueeze(1)
            v_act = out_Q_and_ae[:, i].unsqueeze(1)

            # print(v_act_source)
            # print("v_act_s.shape",v_act_s.shape)
            Q_a = torch.cat((v_act, v_act_s), dim=1)
            # print("Q_a.shape",Q_a.shape)
            v_Q_a = self.activation(self.actions[i](Q_a))
            # print("v_Q_a.shape",v_Q_a.shape)
            rez.append(v_Q_a)
            # exit()

        rez = torch.cat(rez, dim=1)

        # print(rez)
        rez = rez.view(rez.size(0), -1)
        # exit()
        return rez


class TDQN:
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
                 transfer_module=None,
                 writer=None,
                 transfer_param_init=None,
                 **kwargs):
        if transfer_param_init is None:
            self.transfer_param_init = {"w": np.random.random_sample(1)[0], "b": np.random.random_sample(1)[0]}
        else:
            self.transfer_param_init = transfer_param_init
        self.weights_over_time = []
        self.biais_over_time = []
        self.best_fit_over_time = []
        self.ae_errors_over_time = []
        self.p_over_time = []
        self.writer = writer
        self.tranfer_module = transfer_module
        self.device = device
        self.size_mini_batch = batch_size_experience_replay
        self.gamma = gamma
        self.TARGET_UPDATE = target_update
        self.workspace = workspace
        self.policy_net = policy_network.to(self.device)
        self.transfer_net = TranferNetwork(self.policy_net).to(self.device)
        self.memory = Memory()
        self.i_episode = 0
        self.n_actions = self.policy_net.predict.out_features
        self.loss_function = loss_fonction_factory(loss_function)
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer
        self.optimizer = None
        self.reset()

    def save(self, path=None):
        import os
        makedirs(os.path.dirname(path))
        if path is None:
            path = self.workspace / "dqn.pt"
        logger.info("saving dqn at {}".format(path))
        torch.save(self.policy_net, path)
        return path

    def reset(self, reset_weight=True):
        self.memory.reset()
        if reset_weight:
            self.policy_net.reset()

        # if self.tranfer_module is not None and self.tranfer_module.is_q_transfering():
        #     self.transfer_net = self.tranfer_module.get_Q_source()
        # self.weight_transfer = torch.Tensor([self.transfer_param_init["w"]]).to(self.device)
        # self.weight_transfer.requires_grad_()
        # self.biais_transfer = torch.Tensor([self.transfer_param_init["b"]]).to(self.device)
        # self.biais_transfer.requires_grad_()
        # self.parameters = list(self.policy_net.parameters()) + list(
        #     self.transfer_net.parameters())  # + [self.weight_transfer, self.biais_transfer]

        self.optimizer_transfer = optimizer_factory(self.optimizer_type,
                                           self.transfer_net.parameters(),
                                           self.lr,
                                           self.weight_decay)
        self.optimizer_classic = optimizer_factory(self.optimizer_type,
                                           self.policy_net.parameters(),
                                           self.lr,
                                           self.weight_decay)

        self.i_episode = 0
        self.no_need_for_transfer_anymore = False

        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def _optimize_model(self):

        transitions = self.memory.sample(
            len(self.memory) if self.size_mini_batch > len(self.memory) else self.size_mini_batch)

        if self.tranfer_module is not None:
            # reeavalute errors
            self.tranfer_module.evaluate()

            # transfert samples
            if self.tranfer_module.is_experience_replay_transfering():
                size_transfer = self.size_mini_batch - len(transitions)
                if size_transfer > 0 and not self.no_need_for_transfer_anymore:
                    transfer_transitions = self.tranfer_module.get_experience_replay_source().sample(size_transfer)
                    transitions = transitions + transfer_transitions
                self.no_need_for_transfer_anymore = size_transfer <= 0

            # transfer Q
            if self.tranfer_module.is_q_transfering():
                self.transfer_net.set_Q_source(self.tranfer_module.get_Q_source(), self.tranfer_module.get_error())
                self.best_fit_over_time.append(self.tranfer_module.best_fit)
                self.ae_errors_over_time.append(self.tranfer_module.get_error())
                if self.writer is not None:
                    self.writer.add_scalar('ae_best_fit/episode', self.tranfer_module.best_fit, self.i_episode)
                    self.writer.add_scalar('ae_error/episode', self.tranfer_module.get_error(), self.i_episode)
        batch = TransitionGym(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)),
                                      device=self.device,
                                      dtype=torch.uint8)
        nf_ns = [s for s in batch.s_ if s is not None]

        state_batch = torch.cat(batch.s)
        action_batch = torch.cat(batch.a)
        action_batch = action_batch.unsqueeze(1)
        r_batch = torch.cat(batch.r_)

        Q = self.policy_net(state_batch)
        sa_values = Q.gather(1, action_batch)

        Q_transfer = self.transfer_net(state_batch)
        sa_values_t = Q_transfer.gather(1, action_batch)

        ns_values = torch.zeros(len(transitions), device=self.device)
        ns_values_t = torch.zeros(len(transitions), device=self.device)
        if nf_ns:
            ns_values[non_final_mask] = self.target_net(torch.cat(nf_ns)).max(1)[0].detach()
            ns_values_t[non_final_mask] = self.transfer_net(torch.cat(nf_ns)).max(1)[0].detach()
        else:
            logger.warning("Pas d'état non terminaux")

        bootstrap = r_batch + self.gamma * ns_values
        bootstrap_t = r_batch + self.gamma * ns_values_t
        loss_classic = self.loss_function(sa_values, bootstrap.unsqueeze(1))
        loss_transfer = self.loss_function(sa_values_t, bootstrap_t.unsqueeze(1))


        for param in self.transfer_net.Q.parameters():
            param.requires_grad = False
        for param in self.transfer_net.Q_source.parameters():
            param.requires_grad = False
        # print(self.transfer_net)

        self.optimizer_transfer.zero_grad()
        loss_transfer.backward(retain_graph=True)
        for param in self.transfer_net.parameters():
            if param.requires_grad:
                param.grad.data.clamp_(-1, 1)
        self.optimizer_transfer.step()

        for param in self.transfer_net.Q.parameters():
            param.requires_grad = True

        self.optimizer_classic.zero_grad()
        loss_classic.backward()
        for param in self.policy_net.parameters():
            if param.requires_grad:
                param.grad.data.clamp_(-1, 1)
        self.optimizer_classic.step()



    def pi(self, state, action_mask):

        with torch.no_grad():
            if not type(action_mask) == type(np.zeros(1)):
                action_mask = np.asarray(action_mask)
            action_mask[action_mask == 1.] = np.infty
            action_mask = torch.tensor([action_mask], device=self.device, dtype=torch.float)
            s = torch.tensor([state], device=self.device, dtype=torch.float)
            v = self.transfer_net(s)
            # v = self.policy_net(s)
            a = v.squeeze().sub(action_mask).max(1)[1].view(1, 1).item()
            return a

    def update(self, *sample):
        state, action, reward, next_state, done, info = sample
        state = torch.tensor([state], device=self.device, dtype=torch.float)
        if not done:
            next_state = torch.tensor([next_state], device=self.device, dtype=torch.float)
        else:
            next_state = None
        action = torch.tensor([action], device=self.device, dtype=torch.long)
        reward = torch.tensor([float(reward)], device=self.device, dtype=torch.float)
        # print("next_state !!!! ",next_state)
        t = state, action, reward, next_state, done, info
        self.memory.push(*t)
        if self.tranfer_module is not None:
            self.tranfer_module.push(*t)
        self._optimize_model()
        if done:
            self.i_episode += 1
            if self.i_episode % self.TARGET_UPDATE == 0:
                logger.info("[update][i_episode={}] copying weights to target network".format(self.i_episode))
                self.target_net.load_state_dict(self.policy_net.state_dict())
