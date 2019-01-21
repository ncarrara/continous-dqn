# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt

from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.utils_rl.transition.transition import Transition
from ncarrara.utils_rl.visualization.toolsbox import create_Q_histograms, create_Q_histograms_for_actions, \
    fast_create_Q_histograms_for_actions, plot
import logging


class NetFTQ(torch.nn.Module):
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
        super(NetFTQ, self).__init__()
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


class PytorchFittedQ:
    ALL_BATCH = "ALL_BATCH"
    ADAPTATIVE = "ADAPTATIVE"

    def __init__(self,
                 policy_network,
                 device,

                 optimizer=None,
                 loss_function=None,
                 max_ftq_epoch=np.inf,
                 max_nn_epoch=1000,
                 gamma=0.99,
                 learning_rate=0.001,
                 weight_decay=0.001,
                 reset_policy_each_ftq_epoch=True,
                 delta_stop=0,
                 batch_size_experience_replay=50,
                 nn_loss_stop_condition=0.0,
                 disp_states=[],
                 workspace="tmp",
                 action_str=None,
                 test_policy=None
                 ):
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.test_policy = test_policy
        self.action_str = action_str
        self.workspace = workspace
        self.nn_stop_loss_condition = nn_loss_stop_condition
        self.batch_size_experience_replay = batch_size_experience_replay
        self.delta_stop = delta_stop
        self._policy_network = policy_network.to(self.device)
        self._max_ftq_epoch = max_ftq_epoch
        self._max_nn_epoch = max_nn_epoch
        self._gamma = gamma
        self.reset_policy_each_ftq_epoch = reset_policy_each_ftq_epoch
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = optim.RMSprop(params=self._policy_network.parameters(), weight_decay=weight_decay)
        elif self.optimizer == "ADAM":
            self.optimizer = optim.Adam(params=self._policy_network.parameters(),
                                        lr=learning_rate,
                                        weight_decay=weight_decay)
        elif self.optimizer == "RMS_PROP":
            self.optimizer = optim.RMSprop(params=self._policy_network.parameters(),
                                           weight_decay=weight_decay)
        else:
            raise Exception("Unknown optimizer")
        self.loss_function = loss_function
        if self.loss_function == "l1":
            self.loss_function = F.smooth_l1_loss
        elif self.loss_function == "l2":
            self.loss_function = F.mse_loss
        else:
            raise Exception("unknow loss {}".format(self.loss_function))
        self.disp_states = disp_states
        self.statistiques = None
        self.memory = Memory(class_transition=Transition)
        self.reset()

    def reset(self, reset_weight=True):
        self.memory.reset()
        if reset_weight:
            self._policy_network.reset()
        self._id_ftq_epoch = None
        self._non_final_mask = None
        self._non_final_next_states = None
        self._state_batch = None
        self._action_batch = None
        self._reward_batch = None

    def _construct_batch(self, transitions):
        for t in transitions:
            state = torch.tensor([[t.s]], device=self.device, dtype=torch.float)
            if t.s_ is not None:
                next_state = torch.tensor([[t.s_]], device=self.device, dtype=torch.float)
            else:
                next_state = None
            action = torch.tensor([[t.a]], device=self.device, dtype=torch.long)
            reward = torch.tensor([t.r_], device=self.device)
            self.memory.push(state, action, reward, next_state)

        zipped = Transition(*zip(*self.memory.memory))
        state_batch = torch.cat(zipped.s)
        mean = torch.mean(state_batch, 0)
        std = torch.std(state_batch, 0)
        self._policy_network.set_normalization_params(mean, std)

    def _sample_batch(self):
        if self.batch_size_experience_replay == self.ADAPTATIVE:
            size_batch = len(self.memory) / 10
        elif self.batch_size_experience_replay == self.ALL_BATCH:
            size_batch = len(self.memory)
        else:
            size_batch = self.batch_size_experience_replay
        transitions = self.memory.sample(size_batch)
        self.batch_size = len(transitions)
        batch = Transition(*zip(*transitions))

        self._non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)),
                                            device=self.device,
                                            dtype=torch.uint8)
        self._non_final_next_states = torch.cat([s for s in batch.s_ if s is not None])
        self._state_batch = torch.cat(batch.s)
        self._action_batch = torch.cat(batch.a)
        self._reward_batch = torch.cat(batch.r_)

    def q(self, s):
        # feat = [1.00, 1.00, 1.00, 1.0, 0.0, 0.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        # import torch
        q = self._policy_network(torch.tensor([[s]], device=self.device, dtype=torch.float))
        return q.detach().numpy()

    def fit(self, transitions):
        self._construct_batch(transitions)
        self._policy_network.reset()
        self.delta = np.inf
        self._id_ftq_epoch = 0
        rewards=[]
        returns=[]
        while self._id_ftq_epoch < self._max_ftq_epoch and self.delta > self.delta_stop:
            self.logger.info("[epoch_ftq={}] ---------".format(self._id_ftq_epoch))
            self._sample_batch()
            self.logger.info("[epoch_ftq={}] #batch={}".format(self._id_ftq_epoch, len(self._state_batch)))
            losses = self._ftq_epoch()

            self.logger.info("loss {}".format(losses[-1]))

            if self.logger.getEffectiveLevel() is logging.INFO and self.test_policy is not None:
                title="epoch={}.png".format(self._id_ftq_epoch)
                stats = self.test_policy(self.construct_pi(self._policy_network))
                rewards.append(stats[0])
                returns.append(stats[2])
                plt.clf()
                plt.plot(range(len(rewards)), rewards, label="reward",marker='o')
                plt.plot(range(len(returns)), returns, label="returns",marker='o')
                plt.legend()
                plt.title(title)
                plt.grid()
                plt.savefig(self.workspace + '/' + title)
                plt.show()
                plt.close()


            self._id_ftq_epoch += 1
            self.logger.info("[epoch_ftq={}] delta={}".format(self._id_ftq_epoch, self.delta))

        final_network = copy.deepcopy(self._policy_network)

        for state in self.disp_states:
            self.logger.info("Q({})={}".format(state, self._policy_network(
                torch.tensor([[state]], device=self.device, dtype=torch.float)).cpu().detach().numpy()))

        pi = self.construct_pi(final_network)

        return pi

    def construct_pi(self,network):
        def pi(state, action_mask):
            with torch.no_grad():
                if not type(action_mask) == type(np.zeros(1)):
                    action_mask = np.asarray(action_mask)
                action_mask[action_mask == 1.] = np.infty
                action_mask = torch.tensor([action_mask], device=self.device, dtype=torch.float)
                s = torch.tensor([[state]], device=self.device, dtype=torch.float)
                a = network(s).sub(action_mask).max(1)[1].view(1, 1).item()

                return a
        return pi

    def _ftq_epoch(self):
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if self._id_ftq_epoch > 0:
            next_state_values[self._non_final_mask] = self._policy_network(self._non_final_next_states).max(1)[
                0].detach()
        else:
            # la Q function doit retourner zéro (on pourra retourner des valeurs random vu que c'est point fixe, mais ca va ralentir la convergence)
            pass

        self.expected_state_action_values = (next_state_values * self._gamma) + self._reward_batch

        losses = self._optimize_model()

        if self.logger.getEffectiveLevel() is logging.INFO:
            self.logger.info("Creating histograms ...")
            QQ = self._policy_network(self._state_batch)
            state_action_rewards = QQ.gather(1, self._action_batch)
            create_Q_histograms(title="Q(s)_pred_target_e={}".format(self._id_ftq_epoch),
                                values=[self.expected_state_action_values.cpu().numpy(),
                                        state_action_rewards.cpu().detach().numpy().flat],
                                path=self.workspace,
                                labels=["target", "prediction"],
                                inf=-2, sup=2)

            mask_action = np.zeros(len(QQ[0]))
            fast_create_Q_histograms_for_actions(title="actions_Q(s)_pred_target_e={}".format(self._id_ftq_epoch),
                                            QQ=QQ.cpu().detach().numpy(),
                                            path=self.workspace,
                                            labels=self.action_str,
                                            mask_action=mask_action,
                                                 inf=-2, sup=2)

        return losses

    def _optimize_model(self):
        self.delta = self._compute_loss().item()
        if self.reset_policy_each_ftq_epoch:
            self._policy_network.reset()
        stop = False
        nn_epoch = 0
        losses = []
        # torch.set_grad_enabled(True)
        rewards=[]
        returns=[]
        while not stop:
            loss = self._gradient_step()
            losses.append(loss)
            if nn_epoch%500==0:
                self.logger.info("[epoch_ftq={:02}][epoch_nn={:03}] loss={:.4f}"
                                 .format(self._id_ftq_epoch, nn_epoch, loss))
                if self.logger.getEffectiveLevel() is logging.INFO and self.test_policy is not None:
                    stats = self.test_policy(self.construct_pi(self._policy_network))
                    rewards.append(stats[0])
                    returns.append(stats[2])
            cvg = loss < self.nn_stop_loss_condition
            if cvg:
                self.logger.info(
                    "[epoch_ftq={:02}][epoch_nn={:03}] early stopping [loss={}]".format(self._id_ftq_epoch, nn_epoch,
                                                                                        loss))

            nn_epoch += 1
            stop = nn_epoch > self._max_nn_epoch or cvg

        if self.logger.getEffectiveLevel() is logging.INFO and self.test_policy is not None and len(rewards)>0:
            plt.clf()
            title="neural_network_optimisation_epoch_ftq={}".format(self._id_ftq_epoch)
            plt.plot(np.asarray(range(len(rewards)))*100, rewards, label="reward",marker='o')
            plt.plot(np.asarray(range(len(rewards)))*100, returns, label="returns",marker='o')
            plt.title(title)
            plt.grid()
            plt.savefig(self.workspace + '/' + title)
            plt.show()
            plt.close()

        # torch.set_grad_enabled(False)
        self.logger.info("[epoch_ftq={:02}][epoch_nn={:03}] loss={:.4f}"
                             .format(self._id_ftq_epoch, nn_epoch, loss))
        if self.logger.getEffectiveLevel() is logging.INFO:
            plot(losses,
                 title="losses_epoch={}".format(self._id_ftq_epoch),
                 path_save=self.workspace + "/losses_epoch={}".format(self._id_ftq_epoch))
        return losses

    def _compute_loss(self):

        pp = self._policy_network(self._state_batch)
        state_action_values = pp.gather(1, self._action_batch)
        Y_pred = state_action_values
        Y_target = self.expected_state_action_values.unsqueeze(1)
        loss = self.loss_function(Y_pred, Y_target)
        return loss

    def _gradient_step(self):
        loss = self._compute_loss()
        self.optimizer.zero_grad()
        loss.backward()
        for param in self._policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()