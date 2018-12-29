# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import os
from bftq_pydial.algorithms.pytorch_fittedq import ReplayMemory
import bftq_pydial.algorithms.concave_utils as concave_utils
from bftq_pydial.tools.configuration import DEVICE
OptimalPolicy = namedtuple('OptimalPolicy',
                           ('id_action_inf', 'id_action_sup', 'proba_inf', 'proba_sup', 'budget_inf', 'budget_sup'))

TransitionBFTQ = namedtuple('TransitionBFTQ',
                            ('state', 'action', 'next_state', 'reward', 'constraint', 'beta', "hull_id"))


def compute_repartition_value(values, min, max, nb_steps):
    raise Exception("A debugger, il a une erreur, un décalage de une step avant zéro pour les valeurs de zéro")
    # print values
    repartition = np.zeros(nb_steps + 2)
    occ_inf = 0.
    occ_sup = 0.
    for value in values:
        if value < min:
            occ_inf += 1.
        elif value > max:
            occ_sup += 1.
        else:
            pos = np.abs(value - min)
            percent = pos / np.abs(max - min)
            index = percent * float(nb_steps - 1)
            final_index = 1 + int(np.floor(index))
            # print value,"->",pos,percent,index,final_index
            repartition[final_index] += 1
    repartition[0] = occ_inf
    repartition[-1] = occ_sup
    # print repartition
    # exit()
    return repartition


def print_repartitions(repartitions, min, max, steps, epoch):
    s = ""
    for i in range(steps):
        s = s + " {:6.2f}".format(min + i * ((max - min) / float(steps)))
    s = "   x<{:6.2f}".format(min) + s + " {:6.2f}".format(max) + "<x\n"
    for epoch in range(epoch + 1):
        # print repartition
        s += "{:02}  ".format(epoch) + print_repartition(repartitions[epoch], steps) + "\n"
    return s


def print_repartition(repartition, steps):
    sum = np.sum(repartition)
    s = ""
    for ii in range(steps + 2):
        s += " {:6.2f}".format(np.nan if sum == 0 else repartition[ii] * 100. / sum)
    return s


def print_Qr_Qc(epoch, repartitions_Qr, repartitions_Qc, values_Qr, values_Qc, min, max, steps):
    repartitions_Qr[epoch] = compute_repartition_value(values_Qr, min, max, steps)
    repartitions_Qc[epoch] = compute_repartition_value(values_Qc, min, max, steps)

    s = "\nRepartition de la Qr" + "\n"
    s += print_repartitions(repartitions_Qr, min, max, steps, epoch)
    s += "\nRepartition de la Qc" + "\n"
    s += print_repartitions(repartitions_Qc, min, max, steps, epoch)
    return s


def create_Q_histograms(title, values, path, labels):
    plt.clf()
    maxfreq = 0.
    # for i, value in enumerate(values):
    #     n, bins, patches = plt.hist(x=value, bins='auto', label=labels[i], alpha=0.5,rwidth=1.)  # , alpha=0.7, rwidth=0.85)
    #     print i
    n, bins, patches = plt.hist(x=values, bins=np.linspace(-1.5, 1.5, 100), label=labels, alpha=1.,
                                stacked=False)  # , alpha=0.7, rwidth=0.85)
    # maxfreq=np.max([n[i].max() for i in range(len(n))])
    # max = n.max()
    # maxfreq = max if max > maxfreq else maxfreq
    plt.grid(axis='y', alpha=0.75)

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend(loc='upper right')

    # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(path + "/" + title)
    # plt.show()
    plt.close()


def optimal_pia_pib(beta, hull, statistic):
    # statistic["len_hull"] = len(hull)
    if len(hull) == 0:
        raise Exception("Hull is empty")
    elif len(hull) == 1:
        Qc_inf, Qr_inf, beta_inf, action_inf = hull[0]
        res = OptimalPolicy(action_inf, 0, 1., 0., beta_inf, 0.)
        if beta == Qc_inf:
            status = "exact"
        elif beta > Qc_inf:
            status = "too_much_budget"
        else:
            status = "not_solvable"
    else:
        Qc_inf, Qr_inf, beta_inf, action_inf = hull[0]
        if beta < Qc_inf:
            status = "not_solvable"
            res = OptimalPolicy(action_inf, 0, 1., 0., beta_inf, 0.)
        else:
            founded = False
            for k in range(1, len(hull)):
                Qc_sup, Qr_sup, beta_sup, action_sup = hull[k]
                if Qc_inf == beta:
                    founded = True
                    status = "exact"  # en realité avec Qc_inf <= beta and beta < Qc_sup ca devrait marcher aussi
                    res = OptimalPolicy(action_inf, 0, 1., 0., beta_inf, 0.)
                    break
                elif Qc_inf < beta and beta < Qc_sup:
                    founded = True
                    p = (beta - Qc_inf) / (Qc_sup - Qc_inf)
                    status = "regular"
                    res = OptimalPolicy(action_inf, action_sup, 1. - p, p, beta_inf, beta_sup)
                    break
                else:
                    Qc_inf, Qr_inf, beta_inf, action_inf = Qc_sup, Qr_sup, beta_sup, action_sup
            if not founded:  # we have at least Qc_sup budget
                status = "too_much_budget"
                res = OptimalPolicy(action_inf, 0, 1., 0., beta_inf, 0.)  # action_inf = action_sup, beta_inf=beta_sup
    statistic["status"] = status
    return res


class PytorchBudgetedFittedQ:
    ALL_BATCH = "ALL_BATCH"
    ADAPTATIVE = "ADAPTATIVE"

    _GAMMA = None
    _MAX_FTQ_EPOCH = None
    _MAX_NN_EPOCH = None
    _id_ftq_epoch = None
    _non_final_mask = None
    _non_final_next_states = None
    # _state_batch = None
    _action_batch = None
    _reward_batch = None
    _constraint_batch = None
    _policy_network = None
    _beta_batch = None
    _state_beta_batch = None
    _next_state_batch = None

    def __init__(self,
                 policy_network,
                 betas,
                 betas_for_discretisation,
                 N_actions,
                 actions_str=None,
                 optimizer=None,
                 loss_function=None,
                 loss_function_c=None,
                 max_ftq_epoch=np.inf,
                 max_nn_epoch=1000,
                 gamma=0.99,
                 gamma_c=0.99,
                 learning_rate=0.001, weight_decay=0.001,
                 reset_policy_each_ftq_epoch=True,
                 disp=False,
                 delta_stop=0.,
                 batch_size_experience_replay=50,
                 nn_loss_stop_condition=0.0,
                 weights_losses=[1., 1.],
                 disp_states=[],
                 disp_states_ids=[],
                 workspace="tmp",
                 print_q_function=False,
                 ):
        self.print_q_function = print_q_function
        self.weights_losses = weights_losses
        self.NN_LOSS_STOP_CONDITION = nn_loss_stop_condition
        self.BATCH_SIZE_EXPERIENCE_REPLAY = batch_size_experience_replay
        self.DELTA = delta_stop
        self.disp_states = disp_states
        self.disp_states_ids = disp_states_ids
        self.do_dynamic_disp_state = not self.disp_states

        self.disp = disp
        self.workspace = workspace
        self.N_actions = N_actions
        if actions_str is None:
            self.str_actions = [str(a) for a in range(N_actions)]
        else:
            self.str_actions = actions_str
        self.betas = betas
        self.betas_for_discretisation = betas_for_discretisation
        self._policy_network = policy_network.to(DEVICE)
        self._policy_network.reset()
        self._MAX_FTQ_EPOCH = max_ftq_epoch
        self._MAX_NN_EPOCH = max_nn_epoch
        self._GAMMA_C = gamma_c
        self._GAMMA = gamma
        self.RESET_POLICY_NETWORK_EACH_FTQ_EPOCH = reset_policy_each_ftq_epoch
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
        self.loss_function_c = loss_function_c
        if self.loss_function == "l1":
            self.loss_function = F.smooth_l1_loss
        elif self.loss_function == "l2":
            self.loss_function = F.mse_loss
        else:
            raise Exception("unknow loss {}".format(self.loss_function))

        if self.loss_function_c == "l1":
            self.loss_function_c = F.smooth_l1_loss
        elif self.loss_function_c == "l2":
            self.loss_function_c = F.mse_loss
        else:
            raise Exception("unknow loss_c {}".format(self.loss_function_c))

        self.min_print_q = -0.1
        self.max_print_q = 1
        self.memory = ReplayMemory(1000000, TransitionBFTQ)
        self.step_occ = 11
        if self.disp:
            self.repartitions_c = np.zeros((self._MAX_FTQ_EPOCH, self.step_occ + 2))
            # self.repartitions_c[0] = self.step_occ*np.ones(self.step_occ+2)
            self.repartitions_r = np.zeros((self._MAX_FTQ_EPOCH, self.step_occ + 2))
            # self.repartitions_r[0] = self.step_occ*np.ones(self.step_occ + 2)
            self.repartitions_c_b4_opti = np.zeros((self._MAX_FTQ_EPOCH, self.step_occ + 2))
            self.repartitions_r_b4_opti = np.zeros((self._MAX_FTQ_EPOCH, self.step_occ + 2))
            self.repartitions_c_a4_opti = np.zeros((self._MAX_FTQ_EPOCH, self.step_occ + 2))
            self.repartitions_r_a4_opti = np.zeros((self._MAX_FTQ_EPOCH, self.step_occ + 2))
            # self.path_historgrams = self.workspace + "/histograms"
            # if not os.path.exists(self.path_historgrams): os.mkdir(self.path_historgrams)
        self.reset()
    def reset(self, reset_weight=True):
        self.memory.reset()
        if reset_weight:
            self._policy_network.reset()
        self._id_ftq_epoch = None
        self._non_final_mask = None
        self._non_final_next_states = None
        self._action_batch = None
        self._constraint_batch = None
        self._reward_batch = None
        self._beta_batch = None
        self._state_beta_batch = None
        self._states_for_hulls = None

    def _construct_batch(self, transitions):
        if self.disp: print("[epoch_bftq={:02}][_construct_batch] constructing batch ...".format(self._id_ftq_epoch))
        if self.disp and self.do_dynamic_disp_state:
            self.disp_states_ids = []
            self.disp_next_states_ids = []
            self.disp_next_states = []
            self.disp_states = []
        means = {}
        key_states = set()
        it = 0
        lastkeyid = 0
        hull_keys = {}
        # print "transitions",transitions
        for i_t, t in enumerate(transitions):
            # print t
            hull_key = str(t.next_state)
            if self.disp and it % np.ceil(len(transitions) / 10) == 0: print(
                "[epoch_bftq={:02}][_construct_batch] {} transitions proceeded".format(self._id_ftq_epoch, it))
            if hull_key in hull_keys:
                hull_id = hull_keys[hull_key]
            else:
                hull_id = torch.tensor([[[lastkeyid]]], device=DEVICE, dtype=torch.long)
                lastkeyid += 1
                hull_keys[hull_key] = hull_id
            for beta in self.betas:
                if t.next_state is not None:
                    next_state = torch.tensor([[t.next_state]], device=DEVICE, dtype=torch.float)
                else:
                    next_state = torch.tensor([[[np.nan] * self._policy_network.size_state]], device=DEVICE,
                                              dtype=torch.float)
                action = torch.tensor([[t.action]], device=DEVICE, dtype=torch.long)
                reward = torch.tensor([t.reward], device=DEVICE, dtype=torch.float)
                constraint = torch.tensor([t.constraint], device=DEVICE, dtype=torch.float)
                state = torch.tensor([[t.state]], device=DEVICE, dtype=torch.float)
                beta = torch.tensor([[[beta]]], device=DEVICE, dtype=torch.float)

                self.memory.push(state, action, next_state, reward, constraint, beta, hull_id)

            key = str(t.state) + str(t.action)
            key_states.add(str(t.next_state))
            if self.disp and self.do_dynamic_disp_state:
                self.disp_states_ids.append(len(self.disp_states))
                self.disp_states.append(t.next_state)
                self.disp_next_states_ids.append(len(self.disp_next_states))
                self.disp_next_states.append(t.next_state)

            if key in means:
                means[key] = means[key] + np.array([t.reward, 1, t.constraint, 1])
            else:
                means[key] = np.array([t.reward, 1, t.constraint, 1])

            it += 1

        self.nb_unique_hull_to_compute = lastkeyid
        if self.disp: print("nbhull to compute : {}".format(lastkeyid))

        # for k, v in means.iteritems():
        #     print( "R={:.2f},C={:.2f}".format(v[0] / float(v[1]), v[2] / float(v[3]))
        if self.disp: print(
            "[epoch_bftq={:02}][_construct_batch] Nombre de samples : {}, nombre de couple (s,a) uniques : {}" \
                .format(self._id_ftq_epoch, len(self.memory), len(means)))

        # print self.memory.memory
        zipped = TransitionBFTQ(*zip(*self.memory.memory))
        beta_batch = torch.cat(zipped.beta)
        state_batch = torch.cat(zipped.state)
        state_beta_batch = torch.cat((state_batch, beta_batch), dim=2)
        mean = torch.mean(state_beta_batch, 0)
        std = torch.std(state_beta_batch, 0)
        self._policy_network.set_normalization_params(mean, std)
        if self.disp: print(
            "[epoch_bftq={:02}][_construct_batch] constructiong batch ... end".format(self._id_ftq_epoch))

    def _sample_batch(self):
        if self.disp: print("[epoch_bftq={:02}][_sample_batch] sampling mini batch ...".format(self._id_ftq_epoch))
        if self.BATCH_SIZE_EXPERIENCE_REPLAY == self.ADAPTATIVE:
            self.size_mini_batch = len(self.memory) / 10
        elif self.BATCH_SIZE_EXPERIENCE_REPLAY == self.ALL_BATCH:
            self.size_mini_batch = len(self.memory)
        else:
            self.size_mini_batch = self.BATCH_SIZE_EXPERIENCE_REPLAY
        self.mini_batch = self.memory.sample(self.size_mini_batch)
        np.random.shuffle(self.mini_batch)
        zipped = TransitionBFTQ(*zip(*self.mini_batch))
        self._action_batch = torch.cat(zipped.action)
        self._reward_batch = torch.cat(zipped.reward)
        self._constraint_batch = torch.cat(zipped.constraint)
        if self.disp: print(
            "[epoch_bftq={:02}] nb constraint : {}".format(self._id_ftq_epoch, self._constraint_batch.sum()))
        if self.disp: print("[epoch_bftq={:02}] nb reward : {}".format(self._id_ftq_epoch,
                                                                       self._reward_batch[
                                                                           self._reward_batch >= 1.].sum()))
        self._beta_batch = torch.cat(zipped.beta)
        self._state_batch = torch.cat(zipped.state)
        self._next_state_batch = torch.cat(zipped.next_state)
        self._hull_id_batch = torch.cat(zipped.hull_id)
        self._state_beta_batch = torch.cat((self._state_batch, self._beta_batch), dim=2)
        # print self._state_beta_batch
        # exit()
        if self.disp: print(
            "[epoch_bftq={:02}][_sample_batch] size mini batch ={}".format(self._id_ftq_epoch, self.size_mini_batch))
        if self.disp: print("[epoch_bftq={:02}][_sample_batch] sampling mini batch ... done".format(self._id_ftq_epoch))

    def fit(self, transitions):
        self._id_ftq_epoch = 0
        if self.disp: print("[fit] reseting network ....")
        self._policy_network.reset()

        self._construct_batch(transitions)
        dispstateidsrand = np.random.random_integers(0, len(self.disp_states), 10)
        self.delta = np.inf
        while self._id_ftq_epoch < self._MAX_FTQ_EPOCH and self.delta > self.DELTA:
            if self.disp: print("[epoch_bftq={:02}] ---------".format(self._id_ftq_epoch))
            # print tu.get_gpu_memory_map()
            self._sample_batch()
            if self.disp: print("[epoch_bftq={:02}] #batch={}".format(self._id_ftq_epoch, len(self._state_beta_batch)))
            losses = self._ftq_epoch()
            self._id_ftq_epoch += 1

            if self.disp: print("[epoch_bftq={:02}] delta={}".format(self._id_ftq_epoch, self.delta))
            if self.disp:
                for i_s in dispstateidsrand:
                    state = self.disp_next_states[i_s]
                    id = str(self.disp_next_states_ids[i_s])
                    if state is not None:
                        self.draw_Qr_and_Qc(state, self._policy_network,
                                            "next_state={}_epoch={:03}".format(id, self._id_ftq_epoch))

                        hull = self.convexe_hull(s=torch.tensor([state], device=DEVICE, dtype=torch.float32),
                                                 Q=self._policy_network,
                                                 action_mask=np.zeros(self.N_actions),
                                                 id="next_state_" + id, disp=True)

        final_network = copy.deepcopy(self._policy_network)

        if self.disp:
            for i_s in dispstateidsrand:
                state = self.disp_next_states[i_s]
                id = str(self.disp_next_states_ids[i_s])
                if state is not None:
                    self.draw_Qr_and_Qc(state, self._policy_network,
                                        "next_state={}_epoch={:03}".format(id, self._id_ftq_epoch))

                    hull = self.convexe_hull(s=torch.tensor([state], device=DEVICE, dtype=torch.float32),
                                             Q=final_network,
                                             action_mask=np.zeros(self.N_actions),
                                             id="next_state_" + id, disp=True)
            for i_s in dispstateidsrand:
                # i_s = 52
                state = self.disp_states[i_s]
                id = str(self.disp_states_ids[i_s])
                if state is not None:
                    self.draw_Qr_and_Qc(state, self._policy_network,
                                        "state={}_epoch={:03}".format(id, self._id_ftq_epoch))

                    hull = self.convexe_hull(s=torch.tensor([state], device=DEVICE, dtype=torch.float32),
                                             Q=final_network,
                                             action_mask=np.zeros(self.N_actions),
                                             id="state_final_hulls_" + id, disp=True)

        def pi(state, beta, action_mask):
            hull = self.convexe_hull(s=torch.tensor([state], device=DEVICE, dtype=torch.float32), Q=final_network,
                                     action_mask=action_mask,
                                     id="run_" + str(state), disp=False)
            opt = optimal_pia_pib(beta=beta, hull=hull, statistic={})
            rand = np.random.random()
            a = opt.id_action_inf if rand < opt.proba_inf else opt.id_action_sup
            b = opt.budget_inf if rand < opt.proba_inf else opt.budget_sup
            return a, b

        def pi_tmp(state, beta):
            hull = self.convexe_hull(s=torch.tensor([state], device=DEVICE, dtype=torch.float32), Q=final_network,
                                     action_mask=np.zeros(self.N_actions),
                                     id="run_" + str(state), disp=False)
            opt = optimal_pia_pib(beta=beta, hull=hull, statistic={})
            return opt

        self.pi_tmp = pi_tmp

        return pi

    def _is_terminal_state(self, state):
        isnan = torch.sum(torch.isnan(state)) == self._policy_network.size_state
        return isnan

    def compute_opts(self, hulls):
        next_state_beta = torch.zeros((self.size_mini_batch * 2, 1, self._policy_network.size_state + 1),
                                      device=DEVICE)
        i = 0
        opts = [None] * self.size_mini_batch

        status = {"regular": 0, "not_solvable": 0, "too_much_budget": 0, "exact": 0}
        len_hull = 0
        i_non_terminal = 0
        for _, _, next_state, _, _, beta, hull_id in self.mini_batch:
            # print( i

            stats = {}
            if self.disp and i % np.ceil((len(self.mini_batch) / 5)) == 0: print(
                "[epoch_bftq={:02}][_ftq_epoch] processing optimal pia pib {}".format(self._id_ftq_epoch, i))
            if self._is_terminal_state(next_state):
                pass
            else:
                i_non_terminal += 1
                # if not parralele_computing:
                beta = beta.detach().item()
                opts[i] = optimal_pia_pib(beta, hulls[i], stats)
                len_hull += len(hulls[i])

                status[stats["status"]] += 1
                next_state_beta[i * 2 + 0][0] = torch.cat(
                    (next_state, torch.tensor([[[opts[i].budget_inf]]], device=DEVICE, dtype=torch.float32)),
                    dim=2)
                next_state_beta[i * 2 + 1][0] = torch.cat(
                    (next_state, torch.tensor([[[opts[i].budget_sup]]], device=DEVICE, dtype=torch.float32)),
                    dim=2)

            i += 1
        if self.disp: print("[epoch_bftq={:02}][compute_opts] status : {} for {} transitions. Len(hull)={}".format(
            self._id_ftq_epoch,
            status,
            i_non_terminal,
            len_hull / float(
                i_non_terminal)))
        # print tu.get_gpu_memory_map()
        return opts, next_state_beta

    def compute_next_values(self, Q, opts):
        next_state_rewards = torch.zeros(self.size_mini_batch, device=DEVICE)
        next_state_constraints = torch.zeros(self.size_mini_batch, device=DEVICE)
        i = 0
        if self.disp:
            i_non_terminal = 0
            found = False
            warning_qc_negatif = 0.
            warning_qc__negatif = 0.
            next_state_c_neg = 0.
        for _, _, next_state, _, _, _, hull_id in self.mini_batch:
            if self.disp and i % np.ceil((len(self.mini_batch) / 5)) == 0:
                print("[epoch_bftq={:02}][_ftq_epoch] processing mini batch {}".format(self._id_ftq_epoch, i))
            if self._is_terminal_state(next_state):
                next_state_rewards[i] = 0.
                next_state_constraints[i] = 0.
            else:
                if self.disp: i_non_terminal += 1
                opt = opts[i]
                qr_ = Q[i * 2][opt.id_action_inf]
                qr = Q[i * 2 + 1][opt.id_action_sup]
                qc_ = Q[i * 2][self.N_actions + opt.id_action_inf]
                qc = Q[i * 2 + 1][self.N_actions + opt.id_action_sup]
                if self.disp:
                    if qc < 0:
                        warning_qc_negatif += 1.
                    if qc_ < 0:
                        warning_qc__negatif += 1.

                next_state_rewards[i] = opt.proba_inf * qr_ + opt.proba_sup * qr
                next_state_constraints[i] = opt.proba_inf * qc_ + opt.proba_sup * qc

                if self.disp:
                    if next_state_constraints[i] < 0:
                        next_state_c_neg += 1.
                    found = found or False

            i += 1

        if self.disp:
            print("\n[compute_next_values] Q(s') sur le batch")
            create_Q_histograms("Qr(s')_e={}".format(self._id_ftq_epoch),
                                values=next_state_rewards.cpu().numpy().flat,
                                path=self.workspace + "/histogram",
                                labels=["next value"])
            create_Q_histograms("Qc(s')_e={}".format(self._id_ftq_epoch),
                                values=next_state_constraints.cpu().numpy().flat,
                                path=self.workspace + "/histogram",
                                labels=["next value"])
            print("[WARNING] qc < 0 percentage {:.2f}%".format(warning_qc_negatif / i_non_terminal * 100.))
            print("[WARNING] qc_ < 0 percentage {:.2f}%".format(warning_qc__negatif / i_non_terminal * 100.))
            print("[WARNING] next_state_constraints < 0 percentage {:.2f}%".format(
                next_state_c_neg / i_non_terminal * 100.))

        return next_state_rewards, next_state_constraints

    def _ftq_epoch(self):

        if self._id_ftq_epoch > 0:
            hulls = self.compute_hulls(self._next_state_batch, self._hull_id_batch, self._policy_network, disp=False)
            piapib, next_state_beta = self.compute_opts(hulls)
            # print tu.get_gpu_memory_map()
            torch.set_grad_enabled(False)
            Q_next = self._policy_network(next_state_beta)
            next_state_rewards, next_state_constraints = self.compute_next_values(Q_next, piapib)
        else:
            next_state_rewards = torch.zeros(self.size_mini_batch, device=DEVICE)
            next_state_constraints = torch.zeros(self.size_mini_batch, device=DEVICE)

        expected_state_action_rewards = self._reward_batch + (self._GAMMA * next_state_rewards)
        expected_state_action_constraints = self._constraint_batch + (self._GAMMA_C * next_state_constraints)
        losses = self._optimize_model(expected_state_action_rewards, expected_state_action_constraints)

        if self.disp:
            print("Creating histograms ...")
            QQ = self._policy_network(self._state_beta_batch)
            state_action_rewards = QQ.gather(1, self._action_batch)
            state_action_constraints = QQ.gather(1, self._action_batch + self.N_actions)
            # print(expected_state_action_rewards.cpu().numpy())
            # print(expected_state_action_constraints.cpu().numpy())
            # print(state_action_constraints.cpu().numpy().flat)
            create_Q_histograms(title="Qr(s)_pred_target_e={}".format(self._id_ftq_epoch),
                                values=[expected_state_action_rewards.cpu().numpy(),
                                        state_action_rewards.cpu().numpy().flat],
                                path=self.workspace + "/histogram",
                                labels=["target", "prediction"])
            create_Q_histograms(title="Qc(s)_pred_target_e={}".format(self._id_ftq_epoch),
                                values=[expected_state_action_constraints.cpu().numpy(),
                                        state_action_constraints.cpu().numpy().flat],
                                path=self.workspace + "/histogram",
                                labels=["target", "prediction"])

        return losses

    def _optimize_model(self, expected_state_action_rewards, expected_state_action_constraints):
        self.delta = self._compute_loss(expected_state_action_rewards, expected_state_action_constraints,
                                        with_weight=False).item()
        if self.disp: print("[epoch_bftq={:02}][optimize Q] reset neural network ? {}".format(self._id_ftq_epoch,
                                                                                              self.RESET_POLICY_NETWORK_EACH_FTQ_EPOCH))
        if self.RESET_POLICY_NETWORK_EACH_FTQ_EPOCH:
            self._policy_network.reset()
        stop = False
        nn_epoch = 0
        losses = []
        last_loss = np.inf
        torch.set_grad_enabled(True)
        while not stop:
            loss = self._gradient_step(expected_state_action_rewards, expected_state_action_constraints)
            losses.append(loss)
            if self.disp and (min(last_loss, loss) / max(last_loss, loss) < 0.5 or nn_epoch in [0, 1, 2, 3]):
                print("[epoch_bftq={:02}][epoch_nn={:03}] loss={:.4f}".format(self._id_ftq_epoch, nn_epoch, loss))
            last_loss = loss
            cvg = loss < self.NN_LOSS_STOP_CONDITION
            if self.disp and cvg: print(
                "[epoch_bftq={:02}][epoch_nn={:03}] early stopping [loss={}]".format(self._id_ftq_epoch,
                                                                                     nn_epoch,
                                                                                     loss))
            nn_epoch += 1
            stop = nn_epoch > self._MAX_NN_EPOCH or cvg

        if self.disp and not cvg:
            for i in range(3):
                print("[epoch_bftq={:02}][epoch_nn={:03}] loss={:.4f}".format(self._id_ftq_epoch,
                                                                              nn_epoch - 3 + i, losses[-3 + i]))
        del expected_state_action_rewards
        del expected_state_action_constraints
        torch.set_grad_enabled(False)
        return losses

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

    def _gradient_step(self, expected_state_action_rewards, expected_state_action_constraints):
        loss = self._compute_loss(expected_state_action_rewards, expected_state_action_constraints)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self._policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.detach().item()

    def convexe_hull(self, s, action_mask, Q, id, disp):
        hull, colinearity = concave_utils.compute_interest_points_NN(
            s=s,
            Q=Q,
            action_mask=action_mask,
            betas=self.betas_for_discretisation,
            device=DEVICE,
            disp=disp,
            path=self.workspace,
            id=id)
        return hull

    def compute_hulls(self, states, hull_ids, Q, disp):
        if self.disp: print(("[epoch_bftq={:02}] computing hulls ...".format(self._id_ftq_epoch)))
        hulls = np.array([None] * len(states))
        # hulls_already_computed = {}
        k = 0
        i_computation = 0

        computed_hulls = np.array([None] * self.nb_unique_hull_to_compute)
        for i_s, state in enumerate(states):
            hull_id = hull_ids[i_s]
            hull = computed_hulls[hull_id]
            if hull is None and not self._is_terminal_state(state):

                if self.disp and i_computation % np.ceil(self.nb_unique_hull_to_compute / 5) == 0:
                    print("[epoch_bftq={:02}] hull computed : {}".format(
                        self._id_ftq_epoch,
                        # i_s,
                        i_computation))
                hull = self.convexe_hull(s=state,
                                         action_mask=np.zeros(self.N_actions),
                                         Q=Q,
                                         id=str(state.cpu().detach().numpy()) + "_epoch=" + str(self._id_ftq_epoch),
                                         disp=disp)
                computed_hulls[hull_id] = hull
                i_computation += 1
            hulls[i_s] = hull
        if self.disp:
            print(("[epoch_bftq={:02}] hulls actually computed : {}".format(self._id_ftq_epoch, i_computation)))
            print(("[epoch_bftq={:02}] total hulls (=next_states) : {}".format(self._id_ftq_epoch, len(states))))
            print(("[epoch_bftq={:02}] computing hulls [DONE]".format(self._id_ftq_epoch)))

        return hulls

    def draw_Qr_and_Qc(self, s, Q, id):
        plt.clf()
        actions = range(self.N_actions)  # [2
        if not os.path.exists(self.workspace + "/behavior/"):
            os.makedirs(self.workspace + "/behavior/")
        betas = self.betas_for_discretisation  # np.linspace(0, self.beta_max, 100)
        title = id
        yr = np.zeros((len(betas), self.N_actions))
        yc = np.zeros((len(betas), self.N_actions))
        for idx, beta in enumerate(betas):
            # print( s
            qrqc = Q(torch.tensor([[np.append(s, beta)]], device=DEVICE).float()).cpu().detach().numpy()
            # print( qrqc
            yr[idx] = qrqc[0][:self.N_actions]
            yc[idx] = qrqc[0][self.N_actions:]

        for ia in actions:  # self.N_actions):
            # print( yr
            # print( yr[:,ia]
            plt.plot(betas, yr[:, ia], ls="-", marker='o', markersize=2)
        if self.N_actions < 4:
            plt.legend([self.str_actions[a] for a in actions])
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.title(title)
        plt.xlabel("beta")
        plt.ylabel("Qr")
        plt.savefig(self.workspace + "/behavior/Qr_" + title + ".png")
        plt.close()
        plt.clf()

        for ia in actions:  # range(self.N_actions):
            # print( yr
            # print( yr[:, ia]
            plt.plot(betas, yc[:, ia], ls="-", marker='^', markersize=2)

        if self.N_actions < 4:
            plt.legend([self.str_actions[a] for a in actions])
        plt.title(title)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel("beta")
        plt.ylabel("Qc")
        plt.savefig(self.workspace + "/behavior/Qc_" + title + ".png")
        plt.close()
        plt.clf()
        fig, ax = plt.subplots()
        for ia in actions:  # range(self.N_actions):
            plt.plot(yc[:, ia], yr[:, ia], ls="-", marker='v', markersize=2)
            # for ib, budget in enumerate(betas):
            #     ax.annotate("{:.2f}".format(budget), (yc[ib, ia], yr[ib, ia]))

        if self.N_actions < 4:
            plt.legend([self.str_actions[a] for a in actions])
        plt.title(title)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel("Qc")
        plt.ylabel("Qr")
        plt.savefig(self.workspace + "/behavior/QrQc_" + title + ".png")
        plt.close()


class NetBFTQ(torch.nn.Module):
    def _init_weights(self, m):

        if hasattr(m, 'weight'):
            if self.reset_type == "XAVIER":
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif self.reset_type == "ZEROS":
                torch.nn.init.constant_(m.weight.data, 0.)
            else:
                raise ("fuck off mate !")

    def __init__(self, size_state, size_beta_encoder, layers, activation_type="RELU", normalize=False,
                 reset_type="XAVIER",
                 beta_encoder_type="LINEAR", **kwargs):  # REPEAT
        super(NetBFTQ, self).__init__()
        sizes = layers
        self.beta_encoder_type = beta_encoder_type
        self.size_state = size_state
        self.size_beta_encoder = size_beta_encoder
        if activation_type == "RELU":
            self.activation_type = F.relu
        else:
            raise Exception("Unknow activation_type : {}".format(F.relu))
        self.reset_type = reset_type
        self.normalize = normalize
        self.std = None
        self.mean = None
        self.size_action = sizes[-1] / 2
        layers = []
        if size_beta_encoder > 1:
            if self.beta_encoder_type == "LINEAR":
                self.beta_encoder = torch.nn.Linear(1, size_beta_encoder)
            self.concat_layer = torch.nn.Linear(size_state + size_beta_encoder, sizes[0])
        else:
            module = torch.nn.Linear(size_state + size_beta_encoder, sizes[0])
            layers.append(module)
        for i in range(0, len(sizes) - 2):
            module = torch.nn.Linear(sizes[i], sizes[i + 1])
            layers.append(module)

        self.linears = nn.ModuleList(layers)

        self.predict = torch.nn.Linear(sizes[-2], sizes[-1])

    def set_normalization_params(self, mean, std):
        std[std == 0.] = 1.  # on s'en moque, on divisera 0 par 1.
        self.std = std
        self.mean = mean

    def forward(self, x):

        if self.normalize:  # hasattr(self, "normalize"):
            x = (x - self.mean) / self.std

        if self.size_beta_encoder > 1:
            # print "x",x
            beta = x[:, :, -1]
            if self.beta_encoder_type == "REPEAT":
                beta = beta.repeat(1, self.size_beta_encoder)
            elif self.beta_encoder_type == "LINEAR":
                beta = self.beta_encoder(beta)
            else:
                raise "Unknow encoder type : {}".format(self.beta_encoder_type)
            state = x[:, :, 0:-1][:, 0]
            x = torch.cat((state, beta), dim=1)
            x = self.concat_layer(x)
        elif self.size_beta_encoder == 1:
            pass
        else:
            x = x[:, :, 0:-1]

        for i, layer in enumerate(self.linears):
            x = self.activation_type(layer(x))
        x = self.predict(x)  # linear output

        return x.view(x.size(0), -1)

    def reset(self):
        # if self.reset_type == "XAVIER":
        #     print("Reseting network with random value ...")
        # elif self.reset_type == "ZEROS":
        #     print("Reseting network with zeros ...")
        # else:
        #     raise ("fuck off mate !")
        self.apply(self._init_weights)
