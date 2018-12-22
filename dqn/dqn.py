# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
import torch
from dqn.replay_memory import ReplayMemory
from configuration import DEVICE
from dqn.transition import Transition
import copy
import torch.nn.functional as F


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
        self.policy_net = policy_network.to(DEVICE)
        self.target_net = copy.deepcopy(policy_network)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optimizer
        self.memory = ReplayMemory(10000)
        self.i_episode = 0

        if loss_function == "l2":
            self.loss_function = F.mse_loss
        elif loss_function == "l1":
            self.loss_function = F.l1_loss
        else:
            raise Exception("Unknown loss function : {}".format(loss_function))

        if optimizer == "ADAM":
            self.optimizer = torch.optim.Adam(params=self.policy_net.parameters(),
                                              lr=lr, weight_decay=weight_decay)
        else:
            raise Exception("Unknown optimizer : {}".format(optimizer))

    def reset(self, reset_weight=True):
        self.memory.reset()
        if reset_weight:
            self.policy_net.reset()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.i_episode = 0

    def _optimize_model(self):
        if self.BATCH_SIZE_EXPERIENCE_REPLAY == self.ADAPTATIVE:
            size_batch = len(self.memory) / 10
        elif self.BATCH_SIZE_EXPERIENCE_REPLAY == self.ALL_BATCH:
            size_batch = len(self.memory)
        else:
            size_batch = self.BATCH_SIZE_EXPERIENCE_REPLAY

        if size_batch > len(self.memory):
            size_batch = len(self.memory)

        transitions = self.memory.sample(size_batch)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.s_)), device=DEVICE, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.s_
                                           if s is not None])
        self._state_batch = torch.cat(batch.s)
        self._action_batch = torch.cat(batch.a)
        reward_batch = torch.cat(batch.r_)
        state_action_values = self.policy_net(self._state_batch).gather(1, self._action_batch)
        next_state_values = torch.zeros(size_batch, device=DEVICE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        self.expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        loss = self.loss_function(state_action_values, self.expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def pi(self, state, action_mask):

        with torch.no_grad():
            action_mask[action_mask == 1.] = np.infty
            action_mask = torch.tensor([action_mask], device=DEVICE, dtype=torch.float)
            s = torch.tensor([[state]], device=DEVICE, dtype=torch.float)
            a = self.policy_net(s).sub(action_mask).max(1)[1].view(1, 1).item()
            return a

    def update(self, *sample):
        state, action, reward, next_state = sample
        state = torch.tensor([[state]], device=DEVICE, dtype=torch.float)
        if next_state is not None:
            next_state = torch.tensor([[next_state]], device=DEVICE, dtype=torch.float)
        else:
            next_state = None
        action = torch.tensor([[action]], device=DEVICE, dtype=torch.long)
        reward = torch.tensor([reward], device=DEVICE)
        self.memory.push(state, action, reward, next_state)
        self._optimize_model()
        if next_state is None:

            self.i_episode += 1
            if self.i_episode % self.TARGET_UPDATE == 0:
                print("[update][i_episode={}] copying weights to target network".format(self.i_episode))
                self.target_net.load_state_dict(self.policy_net.state_dict())
