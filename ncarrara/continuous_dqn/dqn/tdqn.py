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

from collections import namedtuple

Batch = namedtuple('Batch', ['state', 'action', 'reward', 'non_final_mask', 'non_final_next_state', 'transitions'])


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
                 loss_function_gate=None,
                 ratio_learn_test=None,
                 workspace=None,
                 lr=None,
                 weight_decay=None,
                 transfer_module=None,
                 writer=None,
                 transfer_param_init=None,
                 feature=None,
                 **kwargs):
        self.feature = feature
        if transfer_param_init is None:
            self.transfer_param_init = {"w": np.random.random_sample(1)[0], "b": np.random.random_sample(1)[0]}
        else:
            self.transfer_param_init = transfer_param_init
        self.ratio_learn_test = ratio_learn_test
        self.loss_function_gate = loss_fonction_factory(loss_function_gate)
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
        self.target_update = target_update
        self.workspace = workspace
        self.full_net = policy_network.to(self.device)
        self.n_actions = self.full_net.predict.out_features
        self.loss_function = loss_fonction_factory(loss_function)
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer
        self.reset()

    def save(self, path=None):
        import os
        makedirs(os.path.dirname(path))
        if path is None:
            path = self.workspace / "dqn.pt"
        logger.info("saving dqn at {}".format(path))
        torch.save(self.full_net, path)
        return path

    def reset(self, reset_weight=True):
        self.i_episode = 0
        self.memory = Memory()

        if reset_weight:
            self.full_net.reset()

        # target net of the main and greedy classic net
        self.full_target_net = copy.deepcopy(self.full_net)
        self.full_target_net.load_state_dict(self.full_net.state_dict())
        self.full_target_net.eval()
        self.parameters_full_net = self.full_net.parameters()
        self.optimizer_full_net = optimizer_factory(
            self.optimizer_type,
            self.parameters_full_net,
            self.lr,
            self.weight_decay)

        if self.tranfer_module is not None:
            self.memory_partial_learn = Memory()
            self.memory_partial_test = Memory()
            # net from source tasks
            self.source_net = self.tranfer_module.get_Q_source()

            # net in order to eval bellman residu on test batch
            self.partial_net = copy.deepcopy(self.full_net)
            self.partial_net.load_state_dict(self.full_net.state_dict())
            self.partial_net.eval()
            self.partial_target_net = copy.deepcopy(self.partial_net)
            self.partial_net.load_state_dict(self.partial_net.state_dict())
            self.partial_net.eval()
            self.parameters_partial_net = self.partial_net.parameters()
            self.optimizer_partial_net = optimizer_factory(
                self.optimizer_type,
                self.parameters_partial_net,
                self.lr,
                self.weight_decay)

            # gate to discriminate the choice betwen the source net and the full net
            # this gate compares the partial net and the source net on a test batch (10% of all the transitions)
            self.weight_transfer = torch.Tensor([self.transfer_param_init["w"]]).to(self.device)
            self.weight_transfer.requires_grad_()
            self.biais_transfer = torch.Tensor([self.transfer_param_init["b"]]).to(self.device)
            self.biais_transfer.requires_grad_()
            self.parameters_gate = [self.weight_transfer, self.biais_transfer]
            self.optimizer_gate = optimizer_factory(
                self.optimizer_type,
                self.parameters_gate,
                self.lr,
                self.weight_decay)

    def _construct_batch(self, memory):
        transitions = memory.sample(
            len(memory) if self.size_mini_batch > len(memory) else self.size_mini_batch)

        b = TransitionGym(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, b.s_)),
                                      device=self.device,
                                      dtype=torch.uint8)
        nf_ns = [s for s in b.s_ if s is not None]

        state_batch = torch.cat(b.s)
        action_batch = torch.cat(b.a)
        action_batch = action_batch.unsqueeze(1)
        r_batch = torch.cat(b.r_)
        batch = Batch(
            state=state_batch,
            action=action_batch,
            reward=r_batch,
            non_final_next_state=nf_ns,
            non_final_mask=non_final_mask,
            transitions=transitions)
        return batch

    def _optimize(self, memory, net, target_net, optimizer, parameters):

        batch = self._construct_batch(memory)

        Q = net(batch.state)
        sa_values = Q.gather(1, batch.action)

        ns_values = torch.zeros(len(batch.transitions), device=self.device)
        if batch.non_final_next_state:
            ns_values[batch.non_final_mask] = target_net(torch.cat(batch.non_final_next_state)).max(1)[0].detach()
        else:
            logger.warning("Pas d'état non terminaux")

        bootstrap = batch.reward + self.gamma * ns_values
        loss_classic = self.loss_function(sa_values, bootstrap.unsqueeze(1))
        optimizer.zero_grad()
        loss_classic.backward()
        for param in parameters:
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    def _optimize_model(self):

        #####################################
        ######### optimize full net #########
        #####################################

        self._optimize(
            self.memory,
            self.full_net,
            self.full_target_net,
            self.optimizer_full_net,
            self.parameters_full_net)

        if self.tranfer_module is not None:

            ########################################
            ######### optimize partial net #########
            ########################################

            if len(self.memory_partial_learn) > 0:
                self._optimize(
                    self.memory_partial_learn,
                    self.partial_net,
                    self.partial_target_net,
                    self.optimizer_partial_net,
                    self.parameters_partial_net)

            ########################################
            ######### optimize  GATE ###############
            ########################################

            test_batch = self._construct_batch(self.memory_partial_test)

            # reeavalute errors
            self.tranfer_module.evaluate()
            self.best_fit_over_time.append(self.tranfer_module.best_fit)
            self.ae_errors_over_time.append(self.tranfer_module.get_error())

            Q_source = self.source_net(test_batch.state)
            Q_partial = self.partial_net(test_batch.state)

            sa_values_s = Q_source.gather(1, test_batch.action)
            ns_values_s = torch.zeros(len(test_batch.transitions), device=self.device)
            sa_values_p = Q_partial.gather(1, test_batch.action)
            ns_values_p = torch.zeros(len(test_batch.transitions), device=self.device)
            if test_batch.non_final_next_state:
                ns_values_s[test_batch.non_final_mask] = \
                    self.source_net(torch.cat(test_batch.non_final_next_state)).max(1)[0].detach()
                ns_values_p[test_batch.non_final_mask] = \
                    self.partial_target_net(torch.cat(test_batch.non_final_next_state)).max(1)[0].detach()
            else:
                logger.warning("Pas d'état non terminaux")

            ae_error = self.tranfer_module.get_error()
            p = torch.sigmoid((self.weight_transfer * ae_error + self.biais_transfer))
            bootstrap_t = test_batch.reward + self.gamma * ((1 - p) * (ns_values_p) + p * (ns_values_s))
            loss_transfer = self.loss_function_gate(
                (sa_values_p) * (1 - p) + p * (sa_values_s),
                (bootstrap_t).unsqueeze(1))

            self.optimizer_gate.zero_grad()
            loss_transfer.backward(retain_graph=True)
            for param in self.parameters_gate:
                param.grad.data.clamp_(-1, 1)
            self.optimizer_gate.step()

            # print("--------------------------------")
            # for i in range(len(Q_source)):
            #     print(
            #         "L({:.2f} , [{:.2f} + {:.2f}]) = {:.2f} (diff={:.2f}) ||| L({:.2f} , [{:.2f} + {:.2f}]) = {:.2f} (diff={:.2f})"
            #             .format(
            #             sa_values_p[i].item(),
            #             test_batch.reward[i].item(),
            #             self.gamma * ns_values_p[i].item(),
            #             self.loss_function(sa_values_p[i], (test_batch.reward[i] + self.gamma * ns_values_p[i])).item(),
            #             torch.nn.functional.l1_loss(sa_values_p[i],
            #                                         (test_batch.reward[i] + self.gamma * ns_values_p[i])).item(),
            #             sa_values_s[i].item(),
            #             test_batch.reward[i].item(),
            #             self.gamma * ns_values_s[i].item(),
            #             self.loss_function(sa_values_s[i], (test_batch.reward[i] + self.gamma * ns_values_s[i])).item(),
            #             torch.nn.functional.l1_loss(sa_values_s[i],
            #                                         (test_batch.reward[i] + self.gamma * ns_values_s[i])).item(),
            #         )
            #     )
            # exit()

            if self.tranfer_module is not None and self.tranfer_module.is_q_transfering():
                self.weights_over_time.append(self.weight_transfer)
                self.biais_over_time.append(self.biais_transfer)
                self.p_over_time.append(p)

            if self.writer is not None:
                self.writer.add_scalar('ae_best_fit/episode', self.tranfer_module.best_fit, self.i_episode)
                self.writer.add_scalar('ae_error/episode', self.tranfer_module.get_error(), self.i_episode)
                bootstrap_s = test_batch.reward + self.gamma * ns_values_s
                bootstrap_p = test_batch.reward + self.gamma * ns_values_p
                l_p = self.loss_function((sa_values_p), (bootstrap_p).unsqueeze(1))
                l_s = self.loss_function((sa_values_s), (bootstrap_s).unsqueeze(1))
                self.writer.add_scalar('error_bootstrap_source/episode', l_s, self.i_episode)
                self.writer.add_scalar('error_bootstrap_partial/episode', l_p, self.i_episode)
                # self.writer.add_scalar('diff/episode', l - l_t, self.i_episode)
                # self.writer.add_scalar('loss_transfer/episode', loss_transfer.item(), self.i_episode)
                # self.writer.add_scalar('loss_classic/episode', loss_classic.item(), self.i_episode)
                # self.writer.add_scalar('ae_weight/episode', self.weight_transfer, self.i_episode)
                # self.writer.add_scalar('ae_biais/episode', self.biais_transfer, self.i_episode)
                self.writer.add_scalar('p/episode', p, self.i_episode)
            # exit()

    def pi(self, state, action_mask):
        state = self.feature(state)
        with torch.no_grad():
            if not type(action_mask) == type(np.zeros(1)):
                action_mask = np.asarray(action_mask)
            action_mask[action_mask == 1.] = np.infty
            action_mask = torch.tensor([action_mask], device=self.device, dtype=torch.float)
            s = torch.tensor([state], device=self.device, dtype=torch.float)
            if self.tranfer_module is not None:
                ae_error = self.tranfer_module.get_error()
                p = torch.sigmoid((self.weight_transfer * ae_error + self.biais_transfer))
                v = (self.full_net(s)) * (1 - p) + p * (self.source_net(s))
            else:
                v = self.full_net(s)
            a = v.squeeze().sub(action_mask).max(1)[1].view(1, 1).item()
            return a

    def update(self, *sample):
        state, action, reward, next_state, done, info = sample
        if self.tranfer_module is not None:
            self.tranfer_module.push(*sample)
        state = self.feature(state)
        next_state = self.feature(next_state)
        state = torch.tensor([state], device=self.device, dtype=torch.float)
        if not done:
            next_state = torch.tensor([next_state], device=self.device, dtype=torch.float)
        else:
            next_state = None
        action = torch.tensor([action], device=self.device, dtype=torch.long)
        reward = torch.tensor([float(reward)], device=self.device, dtype=torch.float)
        t = state, action, reward, next_state, done, info
        self.memory.push(*t)

        if self.tranfer_module is not None:
            rdm = np.random.random_sample(1)[0]
            if len(self.memory_partial_test.memory) == 0 or (rdm < self.ratio_learn_test):
                self.memory_partial_test.push(*t)
            else:
                self.memory_partial_learn.push(*t)

        self._optimize_model()
        if done:
            self.i_episode += 1
            if self.i_episode % self.target_update == 0:
                logger.info("[update][i_episode={}] copying weights to target network".format(self.i_episode))
                self.full_target_net.load_state_dict(self.full_net.state_dict())
                if self.tranfer_module is not None:
                    self.partial_target_net.load_state_dict(self.partial_net.state_dict())
