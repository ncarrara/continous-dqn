# -*- coding: utf-8 -*-
import numpy as np
import torch
import copy

from ncarrara.continuous_dqn.dqn.ae_transfer_module import AutoencoderTransferModule
from ncarrara.continuous_dqn.dqn.bellman_transfer_module import BellmanTransferModule
from ncarrara.utils.color import Color
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
                 id,
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
                 info={},
                 transfer_param_init=None,
                 feature=None,
                 **kwargs):
        self.id = id
        self.info = info
        self.tm = transfer_module
        # self.type_transfer_module = type(transfer_module)
        # print(self.type_transfer_module)
        # exit()
        self.feature = feature
        if transfer_param_init is None:
            self.transfer_param_init = {"w": np.random.random_sample(1)[0], "b": np.random.random_sample(1)[0]}
        else:
            self.transfer_param_init = transfer_param_init
        self.ratio_learn_test = ratio_learn_test
        self.loss_function_gate = loss_fonction_factory(loss_function_gate)
        self.best_fit_over_time = []
        self.ae_errors_over_time = []
        self.error_bootstrap_source = []
        self.error_bootstrap_partial = []
        self.p_over_time = []
        self.writer = writer
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

        if self.tm is not None:
            self.tm.reset()
            self.best_net = self.tm.get_best_Q_source()
            self.previous_diff = -np.inf
            self.previous_idx_best_source = self.tm.idx_best_fit
            logging.info(
                "[INITIAL][i_episode{}] Using {} source cstd={:.2f} {} Q function".format(
                    self.i_episode, Color.BOLD, self.tm.best_source_params()["cstd"], Color.END))
            if self.ratio_learn_test > 0:
                # net in order to eval bellman residu on test batch
                self.memory_partial_learn = Memory()
                self.memory_partial_test = Memory()
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
        else:
            self.best_net = self.full_net

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

        if self.tm is not None:

            ########################################
            ######### optimize partial net #########
            ########################################
            if self.ratio_learn_test > 0:
                if len(self.memory_partial_learn) > 0:
                    self._optimize(
                        self.memory_partial_learn,
                        self.partial_net,
                        self.partial_target_net,
                        self.optimizer_partial_net,
                        self.parameters_partial_net)
            else:
                self.partial_net = self.full_net
                self.partial_target_net = self.full_target_net

    def pi(self, state, action_mask):
        state = self.feature(state)
        with torch.no_grad():
            if not type(action_mask) == type(np.zeros(1)):
                action_mask = np.asarray(action_mask)
            action_mask[action_mask == 1.] = np.infty
            action_mask = torch.tensor([action_mask], device=self.device, dtype=torch.float)
            s = torch.tensor([state], device=self.device, dtype=torch.float)
            if self.tm is not None:
                v = self.best_net(s)
            else:
                v = self.full_net(s)
            a = v.squeeze().sub(action_mask).max(1)[1].view(1, 1).item()
            return a

    def update(self, *sample):
        state, action, reward, next_state, done, info = sample
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

        if self.tm is not None:

            if self.ratio_learn_test > 0:
                rdm = np.random.random_sample(1)[0]
                if len(self.memory_partial_test.memory) == 0 or (rdm < self.ratio_learn_test):
                    self.memory_partial_test.push(*t)
                    self.tm.push(*t)
                else:
                    self.memory_partial_learn.push(*t)

        self._optimize_model()
        if done:
            self.i_episode += 1
            if self.i_episode % self.target_update == 0:
                logger.info("[update][i_episode={}] copying weights to target network".format(self.i_episode))
                self.full_target_net.load_state_dict(self.full_net.state_dict())
                if self.tm is not None:
                    self.partial_target_net.load_state_dict(self.partial_net.state_dict())

                    self.tm.update()

                    with torch.no_grad():
                        evaluation_net = self.partial_target_net
                        # evaluation_net = self.full_net

                        batch = TransitionGym(*zip(*self.memory_partial_test.memory))
                        nf_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)),
                                               device=self.device,
                                               dtype=torch.uint8)
                        nf_ns = [s for s in batch.s_ if s is not None]

                        state_batch = torch.cat(batch.s)
                        action_batch = torch.cat(batch.a)
                        reward_batch = torch.cat(batch.r_)

                        action_batch = action_batch.unsqueeze(1)

                        next_state_values = torch.zeros(len(self.memory_partial_test.memory), device=self.device)

                        Q = evaluation_net(state_batch)
                        state_action_values = Q.gather(1, action_batch)
                        if nf_ns:
                            next_state_values[nf_mask] = evaluation_net(torch.cat(nf_ns)).max(1)[0].detach()
                        else:
                            logger.warning("Pas d'état non terminaux")
                        self.expected_state_action_values = (next_state_values * self.gamma) + reward_batch
                        target_loss = self.loss_function(state_action_values,
                                                         self.expected_state_action_values.unsqueeze(1)).item()

                    source_loss = self.tm.get_best_error()
                    diff = source_loss - target_loss

                    cstd_target = self.info["env"].user_params["cstd"]
                    cstd_source = self.tm.best_source_params()["cstd"]
                    cstd_source_before = self.tm.sources_params[self.previous_idx_best_source]["cstd"]

                    if diff > 0:
                        self.best_net = self.full_net
                        cstd = cstd_target
                        if self.previous_diff < 0:
                            logging.info(
                                "[SWITCH][i_episode={}] switching from {} source cstd={:.2f} {} to {} target cstd={:.2f} {} [TARGET cstd={:.2f}]".format(
                                    self.i_episode, Color.BOLD, cstd_source, Color.END, Color.BOLD, cstd_target,
                                    Color.END,cstd_target))
                    else:
                        self.best_net = self.tm.get_best_Q_source()
                        # self.best_net = self.tm.Q_sources[1]
                        cstd = cstd_source

                        if self.previous_diff > 0:
                            logging.info(
                                "[SWITCH][i_episode{}] switching from {} target cstd={:.2f} {} to {} source cstd={:.2f} {} [TARGET cstd={:.2f}]".format(
                                    self.i_episode, Color.BOLD, cstd_target, Color.END, Color.BOLD, cstd_source,
                                    Color.END,cstd_target))
                        elif self.previous_idx_best_source != self.tm.idx_best_fit:
                            logging.info(
                                "[SWITCH][i_episode{}] switching from {} source cstd={:.2f} {} to {} source cstd={:.2f} {} [TARGET cstd={:.2f}]".format(
                                    self.i_episode, Color.BOLD, cstd_source_before, Color.END, Color.BOLD, cstd_source,
                                    Color.END,cstd_target))

                    self.previous_idx_best_source = self.tm.idx_best_fit
                    self.previous_diff = diff

                    self.writer.add_scalar('cerr_{}/episode'.format(self.id), cstd, self.i_episode)

                    self.writer.add_scalar('diff_loss_{}/episode'.format(self.id), diff, self.i_episode)

                    if self.writer is not None:
                        self.writer.add_scalar('target_loss_{}/episode'.format(self.id), target_loss, self.i_episode)

                        # for param, err in zip(self.tm.sources_params, self.tm.errors):
                        self.writer.add_scalar('source_loss_{}/episode'.format(self.id), source_loss, self.i_episode)
