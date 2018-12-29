# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
from utils_rl.transition.replay_memory import ReplayMemory
from utils_rl.transition.transition import Transition, TransitionGym
from bftq_pydial.tools.configuration import C
from utils_rl.visualization.toolsbox import create_Q_histograms, create_Q_histograms_for_actions


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
        # print(all_layers)
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
        # print "x : ",x
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
                 optimizer=None,
                 loss_function=None,
                 max_ftq_epoch=np.inf,
                 max_nn_epoch=1000,
                 gamma=0.99,
                 learning_rate=0.001, weight_decay=0.001,
                 reset_policy_each_ftq_epoch=True,
                 delta_stop=0,
                 batch_size_experience_replay=50,
                 nn_loss_stop_condition=0.0,
                 disp=True,
                 disp_states=[],
                 workspace="tmp",

                 action_str=None,
                 process_between_epoch=None
                 ):
        self.process_between_epoch = process_between_epoch
        self.action_str = action_str
        self.workspace = workspace
        self.nn_stop_loss_condition = nn_loss_stop_condition
        self.batch_size_experience_replay = batch_size_experience_replay
        self.delta_stop = delta_stop
        self._policy_network = policy_network.to(C.device)
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
        self.disp = disp
        self.statistiques = None
        self.memory = ReplayMemory(10000, Transition)
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
            state = torch.tensor([[t.s]], device=C.device, dtype=torch.float)
            if t.s_ is not None:
                next_state = torch.tensor([[t.s_]], device=C.device, dtype=torch.float)
            else:
                next_state = None
            action = torch.tensor([[t.a]], device=C.device, dtype=torch.long)
            reward = torch.tensor([t.r_], device=C.device)
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
                                            device=C.device,
                                            dtype=torch.uint8)
        self._non_final_next_states = torch.cat([s for s in batch.s_ if s is not None])
        self._state_batch = torch.cat(batch.s)
        self._action_batch = torch.cat(batch.a)
        self._reward_batch = torch.cat(batch.r_)

    def fit(self, transitions):
        self._construct_batch(transitions)
        self._policy_network.reset()
        self.delta = np.inf
        self._id_ftq_epoch = 0
        while self._id_ftq_epoch < self._max_ftq_epoch and self.delta > self.delta_stop:
            print("[epoch_ftq={}] ---------".format(self._id_ftq_epoch))
            self._sample_batch()
            print("[epoch_ftq={}] #batch={}".format(self._id_ftq_epoch, len(self._state_batch)))
            losses = self._ftq_epoch()
            print("loss", losses[-1])

            if self.disp and self.process_between_epoch is not None:
                def pi(state, action_mask):
                    action_mask[action_mask == 1.] = np.infty
                    action_mask = torch.tensor([action_mask], device=C.device, dtype=torch.float)
                    s = torch.tensor([[state]], device=C.device, dtype=torch.float)
                    a = self._policy_network(s).sub(action_mask).max(1)[1].view(1, 1).item()
                    return a

                stats = self.process_between_epoch(pi)
                if self._id_ftq_epoch == 0:
                    self.statistiques = stats
                    rewards = self.statistiques[0]
                else:
                    self.statistiques = np.vstack((self.statistiques, stats))
                    rewards = self.statistiques[:, 0]
                plt.clf()
                fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
                ax.plot(range(self._id_ftq_epoch + 1), rewards, label="reward")
                plt.savefig(self.workspace + '/' + "reward_between_epoch.png")
                plt.close()

            self._id_ftq_epoch += 1
            print("[epoch_ftq={}] delta={}".format(self._id_ftq_epoch, self.delta))

        final_network = copy.deepcopy(self._policy_network)

        for state in self.disp_states:
            print("Q({})={}".format(state, self._policy_network(
                torch.tensor([[state]], device=C.device, dtype=torch.float)).cpu().detach().numpy()))

        def pi(state, action_mask):
            action_mask[action_mask == 1.] = np.infty
            action_mask = torch.tensor([action_mask], device=C.device, dtype=torch.float)
            s = torch.tensor([[state]], device=C.device, dtype=torch.float)
            a = final_network(s).sub(action_mask).max(1)[1].view(1, 1).item()
            return a

        return pi

    def _ftq_epoch(self):
        next_state_values = torch.zeros(self.batch_size, device=C.device)
        if self._id_ftq_epoch > 0:
            next_state_values[self._non_final_mask] = self._policy_network(self._non_final_next_states).max(1)[
                0].detach()
        else:
            # la Q function doit retourner zéro (on pourra retourner des valeurs random vu que c'est point fixe, mais ca va ralentir la convergence)
            pass

        self.expected_state_action_values = (next_state_values * self._gamma) + self._reward_batch

        losses = self._optimize_model()

        if self.disp:
            print("Creating histograms ...")
            QQ = self._policy_network(self._state_batch)
            state_action_rewards = QQ.gather(1, self._action_batch)
            create_Q_histograms(title="Q(s)_pred_target_e={}".format(self._id_ftq_epoch),
                                values=[self.expected_state_action_values.cpu().numpy(),
                                        state_action_rewards.cpu().numpy().flat],
                                path=self.workspace + "/histogram",
                                labels=["target", "prediction"])

            mask_action = np.zeros(len(QQ[0]))
            create_Q_histograms_for_actions(title="actions_Q(s)_pred_target_e={}".format(self._id_ftq_epoch),
                                            QQ=QQ.cpu().numpy(),
                                            path=self.workspace + "/histogram",
                                            labels=self.action_str,
                                            mask_action=mask_action)

        return losses

    def _optimize_model(self):
        self.delta = self._compute_loss().item()
        if self.reset_policy_each_ftq_epoch:
            self._policy_network.reset()
        stop = False
        nn_epoch = 0
        losses = []
        torch.set_grad_enabled(True)
        while not stop:
            loss = self._gradient_step()
            losses.append(loss)
            cvg = loss < self.nn_stop_loss_condition
            if cvg:
                print("[epoch_ftq={:02}][epoch_nn={:03}] early stopping [loss={}]".format(self._id_ftq_epoch, nn_epoch,
                                                                                          loss))
            nn_epoch += 1
            stop = nn_epoch > self._max_nn_epoch or cvg
        torch.set_grad_enabled(False)
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
