# -*- coding: utf-8 -*-
import numpy as np
import torch
import copy
import torch.nn.functional as F

from ncarrara.bftq_pydial.bftq.pytorch_budgeted_fittedq import PytorchBudgetedFittedQ
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.utils_rl.transition.transition import TransitionGym
import logging

logger = logging.getLogger(__name__)


class PytorchBudgetedDQN(PytorchBudgetedFittedQ):
    ALL_BATCH = "ALL_BATCH"
    ADAPTATIVE = "ADAPTATIVE"

    def __init__(self,
                 target_update=10,
                 bftq_params=None
                 ):
        # ugly code, but deadline soon ...
        bftq_params["reset_policy_each_ftq_epoch"] = None
        bftq_params["max_ftq_epoch"] = None
        super(PytorchBudgetedDQN, self).__init__(**bftq_params)
        self.TARGET_UPDATE = target_update
        self.target_net = copy.deepcopy(self.policy_network)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.memory = Memory()
        self.i_episode = 0
        self.n_actions = self.policy_net.predict.out_features

    def reset(self, reset_weight=True):
        super(PytorchBudgetedDQN, self).reset(reset_weight=reset_weight)
        self.memory.reset()
        if reset_weight:
            self.policy_net.reset()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.i_episode = 0
        # TDO reset optimizer

    def _optimize_model(self):
        size_batch = self.BATCH_SIZE_EXPERIENCE_REPLAY

        transitions = self.memory.sample(len(self.memory) if size_batch > len(self.memory) else size_batch)

        batch = TransitionGym(*zip(*transitions))

        self._state_batch = torch.cat(batch.s)
        self._next_state_batch = torch.cat(batch.s_)
        self._action_batch = torch.cat(batch.a)
        self._reward_batch = torch.cat(batch.r_)
        self._constraint_batch = torch.cat(batch.c_)
        self._beta_batch = torch.cat(batch.beta)
        self._hull_id_batch = torch.cat(batch.hull_id)

        with torch.no_grad():
            if self._id_ftq_epoch > 0:
                hulls = self.compute_hulls(self._next_state_batch,
                                           self._hull_id_batch,
                                           self._policy_network,
                                           disp=False)
                piapib, next_state_beta = self.compute_opts(hulls)
                Q_next = self._policy_network(next_state_beta)
                next_state_rewards, next_state_constraints = self.compute_next_values(Q_next, piapib)
            else:
                next_state_rewards = torch.zeros(self.size_mini_batch, device=self.device)
                next_state_constraints = torch.zeros(self.size_mini_batch, device=self.device)

            expected_state_action_rewards = self._reward_batch + (self._GAMMA * next_state_rewards)
            expected_state_action_constraints = self._constraint_batch + (self._GAMMA_C * next_state_constraints)

        loss = self._compute_loss(expected_state_action_rewards, expected_state_action_constraints)
        # loss = self.loss_function(state_action_values, self.expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update(self, *sample):
        state, action, reward, next_state, constraint, beta, done, info = sample
        state = torch.tensor([[state]], device=self.device, dtype=torch.float)
        if not done:
            next_state = torch.tensor([[next_state]],
                                      device=self.device,
                                      dtype=torch.float)
        else:
            next_state = torch.tensor([[[np.nan] * self.policy_net.size_state]],
                                      device=self.device,
                                      dtype=torch.float)
        action = torch.tensor([[action]], device=self.device, dtype=torch.long)
        reward = torch.tensor([float(reward)], device=self.device, dtype=torch.float)
        constraint = torch.tensor([float(constraint)], device=self.device, dtype=torch.float)
        beta = torch.tensor([float(beta)], device=self.device, dtype=torch.float)
        self.memory.push(state, action, reward, next_state, constraint, beta, done, info)
        self._optimize_model()
        if next_state is None:
            self.i_episode += 1
            if self.i_episode % self.TARGET_UPDATE == 0:
                logger.info("[update][i_episode={}] copying weights to target network".format(self.i_episode))
                self.target_net.load_state_dict(self.policy_net.state_dict())
