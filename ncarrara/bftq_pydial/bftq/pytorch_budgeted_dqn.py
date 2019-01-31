# -*- coding: utf-8 -*-
import numpy as np
import torch
import copy
import torch.nn.functional as F

from ncarrara.bftq_pydial.bftq.budgeted_utils import BudgetedUtils, TransitionBudgeted
from ncarrara.utils.torch import optimizer_factory
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.utils_rl.transition.transition import TransitionGym
import logging

logger = logging.getLogger(__name__)


class PytorchBudgetedDQN():

    def __init__(self,
                 policy_net,
                 beta_for_discretization,
                 workspace,
                 device,
                 N_action,
                 loss_function,
                 loss_function_c,
                 gamma=0.999,
                 gamma_c=1.0,
                 weights_losses=[1., 1.],
                 target_update=10,
                 size_minibatch=32,
                 state_to_unique_str=lambda s: str(s),
                 action_to_unique_str=lambda a: str(a),
                 ):
        # pattern, par composition (on ne veut pas d héritage pour les algos)
        self.budgeted_utils = BudgetedUtils(beta_for_discretization, workspace, device, N_action, id="NO_ID")
        self.state_to_unique_str = state_to_unique_str
        self.action_to_unique_str = action_to_unique_str
        self.gamma = gamma
        self.gamma_c = gamma_c
        self.size_mini_batch = size_minibatch
        self.target_update = target_update
        self.weights_losses = weights_losses
        self.policy_net = policy_net
        self.memory = Memory(class_transition=TransitionBudgeted)
        self.i_episode = 0
        self.n_actions = self.policy_net.predict.out_features
        self.loss_function_c = loss_function_c
        self.loss_function = loss_function
        self.hulls_key_id = {}
        self.reset()

    def reset(self):
        self.budgeted_utils.reset()
        self.memory.reset()
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optimizer_factory(self.optimizer_type,
                                           self._policy_net.parameters(),
                                           self.learning_rate,
                                           self.weight_decay)
        self.i_episode = 0
        self.current_hull_id = 0
        self.i_update = 0
        self.hull_ids.clear()
        self._id_ftq_epoch = None
        self._non_final_mask = None
        self._non_final_next_states = None
        self._action_batch = None
        self._constraint_batch = None
        self._reward_batch = None
        self._beta_batch = None
        self._state_beta_batch = None
        self._states_for_hulls = None

    # copié coller de BFTQ, pas de raison de mettre ça dans budgeted utils, ca dépend de comment on veut build la loss
    def _compute_loss(self, expected_state_action_rewards, expected_state_action_constraints, with_weight=True):
        QQ = self._policy_network(self._state_beta_batch)
        state_action_rewards = QQ.gather(1, self._action_batch)
        action_batch_qc = self._action_batch + self.N_actions
        state_action_constraints = QQ.gather(1, action_batch_qc)
        loss_Qc = self.loss_function_c(state_action_constraints, expected_state_action_constraints.unsqueeze(1))
        loss_Qr = self.loss_function(state_action_rewards, expected_state_action_rewards.unsqueeze(1))
        w_r, w_c = self.weights_losses
        if with_weight:
            loss = w_c * loss_Qc + w_r * loss_Qr
        else:
            loss = loss_Qc + loss_Qr

        return loss

    def _optimize_model(self):
        transitions = self.memory.sample(
            len(self.memory) if self.size_mini_batch > len(self.memory) else self.size_mini_batch)
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
                next_state_rewards, next_state_constraints = self.budgeted_utils.compute_targets(
                    self.policy_net, self._next_state_batch, self._beta_batch, self._hull_id_batch)
            else:
                next_state_rewards = torch.zeros(self.size_mini_batch, device=self.device)
                next_state_constraints = torch.zeros(self.size_mini_batch, device=self.device)

            expected_state_action_rewards = self._reward_batch + (self._gamma * next_state_rewards)
            expected_state_action_constraints = self._constraint_batch + (self._gamma_c * next_state_constraints)
        loss = self._compute_loss(expected_state_action_rewards, expected_state_action_constraints)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update(self, *sample):
        self.budgeted_utils.change_id(self.i_update)
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

        hull_key = self.state_to_unique_str(next_state)
        if hull_key in self.hulls_key_id:
            hull_id = self.hulls_key_id[hull_key]
        else:
            hull_id = self.current_hull_id
            self.current_hull_id += 1

        self.memory.push(state, action, reward, next_state, constraint, beta, hull_id)
        self._optimize_model()
        if next_state is None:
            self.i_episode += 1
            if self.i_episode % self.target_update == 0:
                logger.info("[update][i_episode={}] copying weights to target network".format(self.i_episode))
                self.target_net.load_state_dict(self.policy_net.state_dict())
        self.i_update += 1
