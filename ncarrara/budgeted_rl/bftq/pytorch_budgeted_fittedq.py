# -*- coding: utf-8 -*-
import gc

import psutil
import sys

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
import matplotlib.pyplot as plt
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import os
from sys import getsizeof

from torch.utils.data import Dataset, ConcatDataset, TensorDataset

from ncarrara.utils.color import Color
from ncarrara.utils.math_utils import update_lims
from ncarrara.utils.torch_utils import optimizer_factory, BaseModule, get_gpu_memory_map, get_memory_for_pid
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.utils_rl.visualization.toolsbox import create_Q_histograms, create_Q_histograms_for_actions, \
    fast_create_Q_histograms_for_actions
import logging

OptimalPolicy = namedtuple('OptimalPolicy',
                           ('id_action_inf', 'id_action_sup', 'proba_inf', 'proba_sup', 'budget_inf', 'budget_sup'))

TransitionBFTQ = namedtuple('TransitionBFTQ',
                            ('state', 'action', 'reward', "next_state", 'constraint', 'beta', "hull_id"))

logger = logging.getLogger(__name__)


def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())


def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)



def compute_interest_points_NN_Qsb(Qsb, action_mask, betas, disp=False, path="tmp", id="default"):
    with torch.no_grad():
        if not type(action_mask) == type(np.zeros(1)):
            action_mask = np.asarray(action_mask)
        # print betas
        N_OK_actions = int(len(action_mask) - np.sum(action_mask))

        dtype = [('Qc', 'f4'), ('Qr', 'f4'), ('beta', 'f4'), ('action', 'i4')]

        if path:
            path = path + "/interest_points/"
        # print path
        colinearity = False
        if disp:
            if not os.path.exists(path):
                os.makedirs(path)

        all_points = np.zeros((N_OK_actions * len(betas), 2))
        all_betas = np.zeros((N_OK_actions * len(betas),))
        all_Qs = np.zeros((N_OK_actions * len(betas),), dtype=int)
        max_Qr = -np.inf
        Qc_for_max_Qr = None
        l = 0
        x = np.zeros((N_OK_actions, len(betas)))
        y = np.zeros((N_OK_actions, len(betas)))
        i_beta = 0
        # ss = s.repeat((len(betas),1,1)) #.squeeze() #torch.full(size=(len(betas)),value=s,device=device)
        # bb = torch.from_numpy(betas).float().unsqueeze(1).unsqueeze(1).to(device=device)
        # sb = torch.cat((ss,bb),dim=2)
        # Qsb = Q(sb)
        for ibeta, beta in enumerate(betas):

            # b_ = torch.tensor([[beta]], device=device, dtype=torch.float32)
            # sb = torch.cat((s.float(), b_), 1)
            # sb = sb.unsqueeze(0)
            # QQ = Q(sb)[0]
            # QQ = QQ.cpu().detach().numpy()
            QQ = Qsb[ibeta]
            for i_a, mask in enumerate(action_mask):
                i_a_ok_act = 0
                if mask == 1:
                    pass
                else:
                    Qr = QQ[i_a]  # TODO c'est peut etre l'inverse ici
                    Qc = QQ[i_a + len(action_mask)]
                    x[i_a_ok_act][i_beta] = Qc
                    y[i_a_ok_act][i_beta] = Qr
                    if Qr > max_Qr:
                        max_Qr = Qr
                        Qc_for_max_Qr = Qc
                    all_points[l] = (Qc, Qr)
                    all_Qs[l] = i_a
                    all_betas[l] = beta
                    l += 1
                    i_a_ok_act += 1

            i_beta += 1

        if disp:
            for i_a in range(0, N_OK_actions):  # len(Q_as)):
                if action_mask[i_a] == 1:
                    pass
                else:
                    plt.plot(x[i_a], y[i_a], linewidth=6, alpha=0.2)
        k = 0
        points = []
        betas = []
        Qs = []
        for point in all_points:
            Qc, Qr = point
            if not (Qr < max_Qr and Qc >= Qc_for_max_Qr):
                # on ajoute que les points non dominés
                points.append(point)
                Qs.append(all_Qs[k])
                betas.append(all_betas[k])
            k += 1
        points = np.array(points)
        if disp:
            # plt.plot(points[:, 0], points[:, 1], '-', linewidth=3, color="tab:cyan")
            plt.plot(all_points[:, 0], all_points[:, 1], 'o', markersize=7, color="blue", alpha=0.1)
            plt.plot(points[:, 0], points[:, 1], 'o', markersize=3, color="red")
            #
            plt.grid()
        try:
            hull = ConvexHull(points)
        except QhullError:
            colinearity = True

        if colinearity:
            idxs_interest_points = range(0, len(points))  # tous les points d'intéret sont bon a prendre
        else:
            stop = False
            max_Qr = -np.inf
            corr_Qc = None
            max_Qr_index = None
            for k in range(0, len(hull.vertices)):
                idx_vertex = hull.vertices[k]
                Qc, Qr = points[idx_vertex]
                if Qr > max_Qr:
                    max_Qr = Qr
                    max_Qr_index = k
                    corr_Qc = Qc
                else:
                    if Qr == max_Qr and Qc < corr_Qc:
                        max_Qr = Qr
                        max_Qr_index = k
                        corr_Qc = Qc
                        k += 1

                    stop = True
                stop = stop or k >= len(hull.vertices)
            idxs_interest_points = []
            stop = False
            # on va itera a l'envers jusq'a ce que Qc change de sens
            Qc_previous = np.inf
            k = max_Qr_index
            j = 0
            # on peut procéder comme ça car c'est counterclockwise
            while not stop:
                idx_vertex = hull.vertices[k]
                Qc, Qr = points[idx_vertex]
                if Qc_previous >= Qc:
                    idxs_interest_points.append(idx_vertex)
                    Qc_previous = Qc
                else:
                    stop = True
                j += 1
                k = (k + 1) % len(hull.vertices)  # counterclockwise
                if j >= len(hull.vertices):
                    stop = True

        if disp:
            plt.title("interest_points_colinear={}".format(colinearity))
            plt.plot(points[idxs_interest_points, 0], points[idxs_interest_points, 1], 'r--', lw=1, color="red")
            plt.plot(points[idxs_interest_points][:, 0], points[idxs_interest_points][:, 1], 'x', markersize=15,
                     color="tab:pink")
            plt.grid()
            plt.savefig(path + str(id) + ".png", dpi=300)
            plt.close()

        rez = np.zeros(len(idxs_interest_points), dtype=dtype)
        k = 0
        for idx in idxs_interest_points:
            Qc, Qr = points[idx]
            beta = betas[idx]
            action = Qs[idx]
            rez[k] = np.array([(Qc, Qr, beta, action)], dtype=dtype)
            k += 1
        if colinearity:
            rez = np.sort(rez, order="Qc")
        else:
            rez = np.flip(rez, 0)  # normalement si ya pas colinearité c'est deja trié dans l'ordre decroissant
        # print rez
        return rez, colinearity  # betas, points, idxs_interest_points, Qs, colinearity


def compute_interest_points_NN(s, Q, action_mask, betas, device, disp=False, path=None, id="default"):
    ss = s.repeat((len(betas), 1, 1))
    bb = torch.from_numpy(betas).float().unsqueeze(1).unsqueeze(1).to(device=device)
    sb = torch.cat((ss, bb), dim=2)
    Qsb = Q(sb)
    return compute_interest_points_NN_Qsb(Qsb, action_mask, betas, disp=disp, path=path, id=id)


def convex_hull(s, action_mask, Q, disp, betas, device, path=None, id="default"):
    if not type(action_mask) == type(np.zeros(1)):
        action_mask = np.asarray(action_mask)
    hull, colinearity = compute_interest_points_NN(
        s=s,
        Q=Q,
        action_mask=action_mask,
        betas=betas,
        device=device,
        disp=disp,
        path=path,
        id=id)
    return hull

def optimal_pia_pib(beta, hull, statistic):
    with torch.no_grad():
        # print(beta)
        # exit()
        # statistic["len_hull"] = len(hull)
        if len(hull) == 0:
            raise Exception("Hull is empty")
        elif len(hull) == 1:
            # print(hull)
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
                    res = OptimalPolicy(action_inf, 0, 1., 0., beta_inf,
                                        0.)  # action_inf = action_sup, beta_inf=beta_sup
        statistic["status"] = status
        return res


class PytorchBudgetedFittedQ:
    ALL_BATCH = "ALL_BATCH"
    ADAPTATIVE = "ADAPTATIVE"

    def __init__(self,
                 policy_network,
                 betas_for_duplication,
                 betas_for_discretisation,
                 device,
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
                 delta_stop=0.,
                 batch_size_experience_replay=50,
                 nn_loss_stop_condition=0.0,
                 weights_losses=[1., 1.],
                 disp_states=[],
                 disp_states_ids=[],
                 workspace="tmp",
                 print_q_function=False,
                 state_to_unique_str=lambda s: str(s),
                 action_to_unique_str=lambda a: str(a),
                 use_data_loader=False,
                 use_data_parallel=False,
                 use_extra_gpu_memory_threshold=0,
                 maximum_number_of_gpu=1

                 ):
        self.memory_tracking = []
        # self.track_memory("init")
        self.state_to_unique_str = state_to_unique_str
        self.action_to_unique_str = action_to_unique_str
        self.device = device
        self.print_q_function = print_q_function
        self.weights_losses = weights_losses
        self.stop_loss_value = nn_loss_stop_condition
        self.BATCH_SIZE_EXPERIENCE_REPLAY = batch_size_experience_replay
        self.DELTA = delta_stop
        self.disp_states = disp_states
        self.disp_states_ids = disp_states_ids
        self.do_dynamic_disp_state = not self.disp_states
        self.workspace = workspace
        self.N_actions = policy_network.predict.out_features // 2
        if actions_str is None:
            actions_str = [str(i) for i in range(self.N_actions)]
        self.actions_str = actions_str

        if type(betas_for_duplication) == type(""):
            self.betas_for_duplication = eval(betas_for_duplication)
        else:
            self.betas_for_duplication = betas_for_duplication
        if type(betas_for_discretisation) == type(""):
            self.betas_for_discretisation = eval(betas_for_discretisation)
        else:
            self.betas_for_discretisation = betas_for_discretisation

        self._policy_network = policy_network
        self.size_state = self._policy_network.size_state
        self.use_data_parallel = use_data_parallel
        self.use_data_loader = use_data_loader
        if self.use_data_loader:
            raise Exception("""Not really working, please stick to basic data loading. 
            Actualy data_loader is useless since the data are already load on gpu (it's not big files or something)""")
        if not self.use_data_parallel:
            self.devices = [self.device]
        else:

            logger.info("Main device : ", self.device)
            # device_ids = [self.device.index]
            device_ids = [self.device.index]
            memory_map = get_gpu_memory_map()
            logger.info("Memory map : {}".format(memory_map))
            for i in range(len(memory_map)):
                if memory_map[i] < use_extra_gpu_memory_threshold \
                        and i != self.device.index \
                        and len(device_ids) < maximum_number_of_gpu:
                    device_ids.append(i)

            logger.info("BFTQ will use those GPU : {}. GPU({}) is the main GPU.".format(device_ids, self.device.index))
            self._policy_network = torch.nn.DataParallel(
                module=self._policy_network,
                output_device=self.device.index,
                device_ids=device_ids)
            self.devices = device_ids

        self._policy_network = self._policy_network.to(self.device)
        # exit()
        self.reset_network()
        self._MAX_FTQ_EPOCH = max_ftq_epoch
        self._MAX_NN_EPOCH = max_nn_epoch
        self._GAMMA_C = gamma_c
        self._GAMMA = gamma
        self.RESET_POLICY_NETWORK_EACH_FTQ_EPOCH = reset_policy_each_ftq_epoch
        self.optimizer_type = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = None
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
        self.reset()
        self.track_memory("__init__ (end)")

    def reset_network(self):
        if self.use_data_parallel:
            self._policy_network.module.reset()
        else:
            self._policy_network.reset()

    def reset(self, reset_weight=True):
        if reset_weight:
            self.reset_network()
        self.optimizer = optimizer_factory(self.optimizer_type,
                                           self._policy_network.parameters(),
                                           self.learning_rate,
                                           self.weight_decay)
        self._id_ftq_epoch = None

    def format_memory(self, memoire):
        for _ in range(len(self.devices) - len(memoire)):
            memoire.append(0)
        accolades = "".join(["{:05} " for _ in range(len(self.devices))])
        accolades = accolades[:-1]
        format = "[m=" + accolades + "]"
        return format.format(*memoire)

    def get_current_memory(self):
        memory = get_memory_for_pid(os.getpid())

        # self.memory_tracking.append(sum)
        return memory

    def track_memory(self, id):
        sum = 0
        for mem in self.get_current_memory():
            sum += mem
        self.memory_tracking.append([id, sum])

    def info(self, message):
        memoire = self.get_current_memory()

        if self._id_ftq_epoch is not None:
            logger.info("[e={:02}]{}{}".format(self._id_ftq_epoch, self.format_memory(memoire), message))
        else:
            logger.info("{}{}".format(self.format_memory(memoire), message))

    def _construct_batch(self, transitions):
        self.track_memory("_construct_batch")
        self.info("[_construct_batch] constructing batch ...")
        memory = Memory(class_transition=TransitionBFTQ)
        if logger.getEffectiveLevel() <= logging.DEBUG and self.do_dynamic_disp_state:
            self.disp_states_ids = []
            self.disp_next_states_ids = []
            self.disp_next_states = []
            self.disp_states = []
            means = {}
        it = 0
        lastkeyid = 0
        hull_keys = {}
        for i_t, t in enumerate(transitions):
            hull_key = self.state_to_unique_str(t.next_state)
            if it % np.ceil(len(transitions) / 10) == 0:
                self.info("[_construct_batch] {} transitions proceeded".format(it))
            if hull_key in hull_keys:
                hull_id = hull_keys[hull_key]
            else:
                hull_id = torch.tensor([[[lastkeyid]]], dtype=torch.long)
                lastkeyid += 1
                hull_keys[hull_key] = hull_id
            if t.next_state is not None:
                next_state = torch.tensor([[t.next_state]], dtype=torch.float)
            else:
                next_state = torch.tensor([[[np.nan] * self.size_state]], dtype=torch.float)
            action = torch.tensor([[t.action]], dtype=torch.long)
            reward = torch.tensor([t.reward], dtype=torch.float)
            constraint = torch.tensor([t.constraint], dtype=torch.float)
            state = torch.tensor([[t.state]], dtype=torch.float)
            if len(self.betas_for_duplication) > 0:
                for beta in self.betas_for_duplication:
                    beta = torch.tensor([[[beta]]], dtype=torch.float)
                    memory.push(state, action, reward, next_state, constraint, beta, hull_id)
            else:
                beta = torch.tensor([[[t.beta]]], dtype=torch.float)
                memory.push(state, action, reward, next_state, constraint, beta, hull_id)

            if logger.getEffectiveLevel() <= logging.DEBUG and self.do_dynamic_disp_state:
                self.disp_states_ids.append(len(self.disp_states))
                self.disp_states.append(t.next_state)
                self.disp_next_states_ids.append(len(self.disp_next_states))
                self.disp_next_states.append(t.next_state)
                key = self.state_to_unique_str(t.state) + self.action_to_unique_str(t.action)
                if key in means:
                    means[key] = means[key] + np.array([t.reward, 1, t.constraint, 1])
                else:
                    means[key] = np.array([t.reward, 1, t.constraint, 1])

            it += 1

        self.nb_unique_hull_to_compute = lastkeyid

        batch = memory.sample(len(memory))
        self.size_batch = len(batch)
        zipped = TransitionBFTQ(*zip(*batch))
        action_batch = torch.cat(zipped.action).to(self.device)
        reward_batch = torch.cat(zipped.reward).to(self.device)
        constraint_batch = torch.cat(zipped.constraint).to(self.device)

        beta_batch = torch.cat(zipped.beta).to(self.device)
        state_batch = torch.cat(zipped.state).to(self.device)
        next_state_batch = torch.cat(zipped.next_state).to(self.device)
        hull_id_batch = torch.cat(zipped.hull_id).to(self.device)
        state_beta_batch = torch.cat((state_batch, beta_batch), dim=2).to(self.device)
        mean = torch.mean(state_beta_batch, 0).to(self.device)
        std = torch.std(state_beta_batch, 0).to(self.device)
        if self.use_data_parallel:
            self._policy_network.module.set_normalization_params(mean, std)
        else:
            self._policy_network.set_normalization_params(mean, std)
        self.info("[_construct_batch] nbhull to compute : {}".format(lastkeyid))
        self.info("[_construct_batch] Nombre de samples : {}".format(len(memory)))
        if logger.getEffectiveLevel() <= logging.DEBUG and self.do_dynamic_disp_state:
            self.info("nombre de couple (s,a) uniques : {}".format(len(means)))
        self.info("[_construct_batch] sum of constraint : {}".format(constraint_batch.sum()))
        self.info("[_construct_batch] nb reward >= 1 : {}".format(reward_batch[reward_batch >= 1.].sum()))
        self.info("[_construct_batch] constructing batch ... end {}")
        self.track_memory("_construct_batch (end)")

        unique_hull_to_compute = []
        hull_ids = []
        for idx_in_batch, hull_id in enumerate(hull_id_batch):
            hull_id = hull_id.item()
            if hull_id not in hull_ids and not self._is_terminal_state(next_state_batch[idx_in_batch]):
                unique_hull_to_compute.append([hull_id, idx_in_batch])
                hull_ids.append(hull_id)
        unique_hull_to_compute = np.array(unique_hull_to_compute)

        return state_beta_batch, state_batch, action_batch, reward_batch, \
               constraint_batch, next_state_batch, hull_id_batch, beta_batch, unique_hull_to_compute

    def getsizeof(self, a, name):
        self.info("size {} : [{}{}{}]".format(name, Color.BOLD, getsizeof(a), Color.END))

    def fit(self, transitions):
        self.track_memory("fit")
        self._id_ftq_epoch = 0
        self.info("[fit] reseting network ...")
        self.reset_network()

        sb_batch, s_batch, a_batch, r_batch, c_batch, ns_batch, h_batch, b_batch, unique_hull_to_compute = self._construct_batch(
            transitions)
        if logger.getEffectiveLevel() <= logging.DEBUG:
            dispstateidsrand = np.random.random_integers(0, len(self.disp_states) - 1, 10)
        self.delta = np.inf
        while self._id_ftq_epoch < self._MAX_FTQ_EPOCH and self.delta > self.DELTA:
            self.track_memory("e={}".format(self._id_ftq_epoch))
            # cpuStats()
            # memReport()
            self.info("-------------------------")
            _ = self._ftq_epoch(sb_batch, a_batch, r_batch, c_batch, ns_batch, h_batch, b_batch, unique_hull_to_compute)
            self.info("delta={}".format(self.delta))
            if logger.getEffectiveLevel() <= logging.DEBUG:
                for i_s in dispstateidsrand:
                    state = self.disp_next_states[i_s]
                    id = self.state_to_unique_str(self.disp_next_states_ids[i_s])
                    if state is not None:
                        self.draw_Qr_and_Qc(state, self._policy_network,
                                            "next_state={}_epoch={:03}".format(id, self._id_ftq_epoch))

                        _ = convex_hull(s=torch.tensor([state], device=self.device, dtype=torch.float32),
                                        Q=self._policy_network,
                                        action_mask=np.zeros(self.N_actions),
                                        id="next_state_" + id, disp=True,
                                        betas=self.betas_for_discretisation,
                                        device=self.device,
                                        path=self.workspace)
                for i_s in dispstateidsrand:
                    # i_s = 52
                    state = self.disp_states[i_s]
                    id = str(self.disp_states_ids[i_s])
                    if state is not None:
                        self.draw_Qr_and_Qc(state, self._policy_network,
                                            "state={}_epoch={:03}".format(id, self._id_ftq_epoch))

                        _ = convex_hull(s=torch.tensor([state], device=self.device, dtype=torch.float32),
                                        Q=self._policy_network,
                                        action_mask=np.zeros(self.N_actions),
                                        id="next_state_" + id, disp=True,
                                        betas=self.betas_for_discretisation,
                                        device=self.device,
                                        path=self.workspace)
            self._id_ftq_epoch += 1

        return self._policy_network

    def empty_cache(self):
        memoire_before = get_memory_for_pid(os.getpid())
        torch.cuda.empty_cache()
        memoire_after = get_memory_for_pid(os.getpid())
        self.info("empty cache {} -> {}".format(self.format_memory(memoire_before), self.format_memory(memoire_after)))

    def _ftq_epoch(self, sb_batch, a_batch, r_batch, c_batch, ns_batch, h_batch, b_batch, unique_hull_to_compute):
        self.track_memory("ftq_epoch")
        self.info("[_ftq_epoch] start ...")
        with torch.no_grad():
            if self._id_ftq_epoch > 0:
                hulls = self.compute_hulls(ns_batch, h_batch, self._policy_network, unique_hull_to_compute)
                piapib, next_state_beta = self.compute_opts(ns_batch, b_batch, h_batch, hulls)
                self.info("Q next")
                self.track_memory("Q_next")
                Q_next = self._policy_network(next_state_beta)
                Q_next = Q_next.detach()
                self.track_memory("Q_next (end)")
                self.info("Q next end")
                self.empty_cache()
                ns_r, ns_c = self.compute_next_values(ns_batch, h_batch, Q_next, piapib)
            else:
                ns_r = torch.zeros(self.size_batch, device=self.device)
                ns_c = torch.zeros(self.size_batch, device=self.device)

            label_r = r_batch + (self._GAMMA * ns_r)
            label_c = c_batch + (self._GAMMA_C * ns_c)
            self.empty_cache()

        losses = self._optimize_model(sb_batch, a_batch, label_r, label_c)

        self.empty_cache()
        if logger.getEffectiveLevel() <= logging.DEBUG:
            with torch.no_grad():
                self.info("Creating histograms ...")
                self.info("forward pass ...")
                QQ = self._policy_network(sb_batch)
                self.info("forward pass ... end")
                state_action_rewards = QQ.gather(1, a_batch)
                state_action_constraints = QQ.gather(1, a_batch + self.N_actions)
                create_Q_histograms(title="Qr(s)_pred_target_e={}".format(self._id_ftq_epoch),
                                    values=[label_r.cpu().numpy(),
                                            state_action_rewards.cpu().numpy().flatten()],
                                    path=self.workspace + "/histogram",
                                    labels=["target", "prediction"])
                create_Q_histograms(title="Qc(s)_pred_target_e={}".format(self._id_ftq_epoch),
                                    values=[label_c.cpu().numpy(),
                                            state_action_constraints.cpu().numpy().flatten()],
                                    path=self.workspace + "/histogram",
                                    labels=["target", "prediction"])

                QQr = QQ[:, 0:self.N_actions]
                QQc = QQ[:, self.N_actions:2 * self.N_actions]
                mask_action = np.zeros(len(self.actions_str))
                fast_create_Q_histograms_for_actions(title="actions_Qr(s)_pred_target_e={}".format(self._id_ftq_epoch),
                                                     QQ=QQr.cpu().numpy(),
                                                     path=self.workspace + "/histogram",
                                                     labels=self.actions_str,
                                                     mask_action=mask_action)
                fast_create_Q_histograms_for_actions(title="actions_Qc(s)_pred_target_e={}".format(self._id_ftq_epoch),
                                                     QQ=QQc.cpu().numpy(),
                                                     path=self.workspace + "/histogram",
                                                     labels=self.actions_str,
                                                     mask_action=mask_action)

        self.empty_cache()
        self.info("[_ftq_epoch] ... end")
        self.track_memory("ftq_epoch (end)")
        return losses

    def save_policy(self, policy_path=None):
        if policy_path is None:
            policy_path = self.workspace + "/policy.pt"
        self.info("saving bftq policy at {}".format(policy_path))
        torch.save(self._policy_network, policy_path)
        return policy_path



    def _is_terminal_state(self, state):
        isnan = torch.sum(torch.isnan(state)) == self.size_state
        return isnan

    def compute_opts(self, ns_batch, b_batch, h_batch, hulls):
        with torch.no_grad():
            self.track_memory("compute_opts")
            self.info("computing ops ... ")
            self.info("before zeros (for next_state_betas)")
            next_state_beta = torch.zeros((self.size_batch * 2, 1, self.size_state + 1),
                                          device=self.device)
            self.info("after zeros (for next_state_betas)")
            i = 0
            opts = [None] * self.size_batch

            status = {"regular": 0, "not_solvable": 0, "too_much_budget": 0, "exact": 0}
            len_hull = 0
            i_non_terminal = 0
            self.info("before zip")
            zipped = zip(ns_batch, b_batch, h_batch)
            self.info("after zip")
            for next_state, beta, hull_id in zipped:

                stats = {}
                if i % np.ceil((self.size_batch / 5)) == 0:
                    self.info("[_ftq_epoch] processing optimal pia pib {}".format(i))
                if self._is_terminal_state(next_state):
                    pass
                else:
                    i_non_terminal += 1
                    beta = beta.detach().item()
                    opts[i] = optimal_pia_pib(beta, hulls[i], stats)
                    len_hull += len(hulls[i])
                    status[stats["status"]] += 1
                    next_state_beta[i * 2 + 0][0] = torch.cat(
                        (next_state, torch.tensor([[opts[i].budget_inf]], device=self.device, dtype=torch.float32)),
                        dim=1)
                    next_state_beta[i * 2 + 1][0] = torch.cat(
                        (next_state, torch.tensor([[opts[i].budget_sup]], device=self.device, dtype=torch.float32)),
                        dim=1)
                i += 1
            self.info("[compute_opts] status : {} for {} transitions. Len(hull)={}".format(status, i_non_terminal,
                                                                                           len_hull / float(
                                                                                               i_non_terminal)))
            self.info("computing opts ... end")
            self.track_memory("compute_opt (end)")
            return opts, next_state_beta

    def compute_next_values(self, ns_batch, h_batch, Q, opts):
        with torch.no_grad():
            self.info("computing next values ...")
            self.track_memory("compute_next_values")
            next_state_rewards = torch.zeros(self.size_batch, device=self.device)
            next_state_constraints = torch.zeros(self.size_batch, device=self.device)
            i = 0
            i_non_terminal = 0
            found = False
            warning_qc_negatif = 0.
            warning_qc__negatif = 0.
            next_state_c_neg = 0.
            for next_state, hull_id in zip(ns_batch, h_batch):
                if i % np.ceil((self.size_batch / 5)) == 0:
                    self.info("[_ftq_epoch] processing mini batch {}".format(i))
                if self._is_terminal_state(next_state):
                    next_state_rewards[i] = 0.
                    next_state_constraints[i] = 0.
                else:
                    i_non_terminal += 1
                    opt = opts[i]
                    qr_ = Q[i * 2][opt.id_action_inf]
                    qr = Q[i * 2 + 1][opt.id_action_sup]
                    qc_ = Q[i * 2][self.N_actions + opt.id_action_inf]
                    qc = Q[i * 2 + 1][self.N_actions + opt.id_action_sup]

                    if qc < 0:
                        warning_qc_negatif += 1.
                    if qc_ < 0:
                        warning_qc__negatif += 1.

                    next_state_rewards[i] = opt.proba_inf * qr_ + opt.proba_sup * qr
                    next_state_constraints[i] = opt.proba_inf * qc_ + opt.proba_sup * qc

                    if next_state_constraints[i] < 0:
                        next_state_c_neg += 1.
                    found = found or False

                i += 1

            if logger.getEffectiveLevel() <= logging.DEBUG:
                self.info("\n[compute_next_values] Q(s') sur le batch")
                create_Q_histograms("Qr(s')_e={}".format(self._id_ftq_epoch),
                                    values=next_state_rewards.cpu().numpy().flatten(),
                                    path=self.workspace + "/histogram",
                                    labels=["next value"])
                create_Q_histograms("Qc(s')_e={}".format(self._id_ftq_epoch),
                                    values=next_state_constraints.cpu().numpy().flatten(),
                                    path=self.workspace + "/histogram",
                                    labels=["next value"])

            self.info("[WARNING] qc < 0 percentage {:.2f}%".format(warning_qc_negatif / i_non_terminal * 100.))
            self.info("[WARNING] qc_ < 0 percentage {:.2f}%".format(warning_qc__negatif / i_non_terminal * 100.))
            self.info("[WARNING] next_state_constraints < 0 percentage {:.2f}%".format(
                next_state_c_neg / i_non_terminal * 100.))
            self.info("compute next values ... end")
            self.track_memory("compute_next_values (end)")
            return next_state_rewards, next_state_constraints

    def _optimize_model(self, sb_batch, a_batch, label_r, label_c):

        sb_batch = sb_batch.to(self.device)
        a_batch = a_batch.to(self.device)
        label_r = label_r.to(self.device)
        label_c = label_c.to(self.device)
        self.info("optimize model ...")
        self.track_memory("delta")
        with torch.no_grad():
            self.info("computing delta ...")
            # no need gradient just for computing delta ....
            if self.use_data_loader:
                logger.warning("--- We are not computing delta ---")
                self.delta = np.inf
            else:
                self.delta = self._compute_loss(sb_batch, a_batch, label_r, label_c, with_weight=False).detach().item()
            self.info("computing delta ... done")
            self.empty_cache()
        self.track_memory("delta (end)")
        self.info("reset neural network ? {}".format(self.RESET_POLICY_NETWORK_EACH_FTQ_EPOCH))
        if self.RESET_POLICY_NETWORK_EACH_FTQ_EPOCH:
            self.reset_network()
        stop = False
        nn_epoch = 0
        losses = []
        last_loss = np.inf
        self.info("gradient descent ...")
        self.track_memory("GD")
        if self.use_data_loader:
            dset = TensorDataset(sb_batch, a_batch, label_r, label_c)
            batch_size = len(sb_batch)
            self.info("Data loader, batch_size={} (len_dataset={})".format(batch_size, len(dset)))
            dataloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=64)

            # while not stop:

            while not stop:
                loss_value = 0.0
                for idata, data in enumerate(dataloader):
                    self.info("".join([str(x.shape) for x in data]))
                    loss = self._gradient_step(*data)
                    loss_value += loss
                    self.info("[epoch_nn={:03}] loss_value={}".format(nn_epoch, loss_value))
                losses.append(loss_value)
                if (min(last_loss, loss_value) / max(last_loss, loss_value) < 0.5 or nn_epoch in [0, 1, 2, 3]):
                    self.info("[epoch_nn={:03}] loss={:.4f}".format(nn_epoch, loss_value))
                last_loss = loss_value
                cvg = loss_value < self.stop_loss_value
                if cvg: self.info("[epoch_nn={:03}] early stopping [loss={}]".format(nn_epoch, loss_value))
                nn_epoch += 1
                stop = nn_epoch > self._MAX_NN_EPOCH or cvg
                if stop:
                    break
        else:
            while not stop:
                loss = self._gradient_step(sb_batch, a_batch, label_r, label_c)
                losses.append(loss)
                if (min(last_loss, loss) / max(last_loss, loss) < 0.5 or nn_epoch in [0, 1, 2, 3]):
                    self.info("[epoch_nn={:03}] loss={:.4f}".format(nn_epoch, loss))
                last_loss = loss
                cvg = loss < self.stop_loss_value
                if cvg:
                    self.info("[epoch_nn={:03}] early stopping [loss={}]".format(nn_epoch, loss))
                nn_epoch += 1
                stop = nn_epoch > self._MAX_NN_EPOCH or cvg
        self.track_memory("GD (end)")
        if not cvg:
            for i in range(3):
                self.info("[epoch_nn={:03}] loss={:.4f}".format(nn_epoch - 3 + i, losses[-3 + i]))
        self.info("gradient descent ... end")
        # del label_r, label_c
        self.empty_cache()
        self.info("optimize model ... done")
        self.track_memory("optimize model (end)")
        return losses

    def _compute_loss(self, sb_batch, a_batch, label_r, label_c, with_weight=True):
        output = self._policy_network(sb_batch)
        state_action_rewards = output.gather(1, a_batch)
        action_batch_qc = a_batch + self.N_actions
        state_action_constraints = output.gather(1, action_batch_qc)

        loss_Qc = self.loss_function_c(state_action_constraints, label_c.unsqueeze(1))
        loss_Qr = self.loss_function(state_action_rewards, label_r.unsqueeze(1))
        w_r, w_c = self.weights_losses
        if with_weight:
            loss = w_c * loss_Qc + w_r * loss_Qr
        else:
            loss = loss_Qc + loss_Qr
        return loss

    def _gradient_step(self, sb_batch, a_batch, label_r, label_c):
        loss = self._compute_loss(sb_batch, a_batch, label_r, label_c)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self._policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.detach().item()


    def compute_hulls(self, ns_batch, h_batch, Q, unique_hull_to_compute):
        with torch.no_grad():
            self.track_memory("compute hulls")
            self.info("computing hulls ...")

            i_computation = 0

            ns_batch_unique = ns_batch[unique_hull_to_compute[:, 1]]
            ss = ns_batch_unique \
                .squeeze() \
                .repeat((1, len(self.betas_for_discretisation))) \
                .reshape(-1) \
                .reshape((len(ns_batch_unique) * len(self.betas_for_discretisation), self.size_state))
            bb = torch.from_numpy(self.betas_for_discretisation).float().unsqueeze(1).to(device=self.device)
            bb = bb.repeat((len(ns_batch_unique), 1))
            sb = torch.cat((ss, bb), dim=1)
            sb = sb.unsqueeze(1)
            Qsb = Q(sb)

            computed_hulls = np.array([None] * len(ns_batch_unique))
            for i_ns_unique in range(len(ns_batch_unique)):
                if i_ns_unique % np.ceil(self.nb_unique_hull_to_compute / 5) == 0:
                    self.info("hull computed : {}".format(i_ns_unique))
                hull, _ = compute_interest_points_NN_Qsb(
                    Qsb[i_ns_unique:i_ns_unique + len(self.betas_for_discretisation)],
                    action_mask=np.zeros(self.N_actions),
                    betas=self.betas_for_discretisation,
                    disp=False,
                    path=self.workspace)
                computed_hulls[i_ns_unique] = hull

            hulls = np.array([None] * len(ns_batch))

            # dict_unique_hll_top_compute = collections.OrderedDict(unique_hull_to_compute)
            hull_ids = unique_hull_to_compute[:, 0]
            for i_ns, state in enumerate(ns_batch):
                hull_id = h_batch[i_ns].item()
                i_ns_unique = np.where(hull_ids == hull_id)
                hull = computed_hulls[i_ns_unique]
                if len(hull) <=0 :
                    pass # ca veut dire que c'est un etat None terminal
                elif len(hull) > 1:
                    raise Exception("Number of unique hulls hull_ids should not be > 1")
                else:
                    hull = hull[0]
                hulls[i_ns] = hull

            self.track_memory("compute_hull (end boucle)")
            self.info("hulls actually computed : {}".format(i_computation))
            self.info("total hulls (=next_states) : {}".format(len(ns_batch)))
            self.info("computing hulls [DONE] ")
            self.empty_cache()
            self.track_memory("compute hulls (end)")
            return hulls

    def draw_Qr_and_Qc(self, s, Q, id):
        with torch.no_grad():
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
                qrqc = Q(torch.tensor([[np.append(s, beta)]], device=self.device).float()).cpu().detach().numpy()
                # print( qrqc
                yr[idx] = qrqc[0][:self.N_actions]
                yc[idx] = qrqc[0][self.N_actions:]

            lims_x = (-1.1, 1.1)
            lims_y = (-1.1, 1.1)
            for ia in actions:
                plt.plot(betas, yr[:, ia], ls="-", marker='o', markersize=2)
                lims_x = update_lims(lims_x, betas)
                lims_y = update_lims(lims_y, yr[:, ia])
            if self.N_actions < 4:
                plt.legend([self.actions_str[a] for a in actions])
            plt.xlim(*lims_x)
            plt.ylim(*lims_y)
            plt.title(title)
            plt.xlabel("beta")
            plt.ylabel("Qr")
            plt.grid()
            plt.savefig(self.workspace + "/behavior/Qr_" + title + ".png")
            plt.close()
            plt.clf()

            lims_x = (-1.1, 1.1)
            lims_y = (-1.1, 1.1)
            for ia in actions:
                plt.plot(betas, yc[:, ia], ls="-", marker='^', markersize=2)
                lims_x = update_lims(lims_x, betas)
                lims_y = update_lims(lims_y, yc[:, ia])
            if self.N_actions < 4:
                plt.legend([self.actions_str[a] for a in actions])
            plt.title(title)
            plt.xlim(*lims_x)
            plt.ylim(*lims_y)
            plt.xlabel("beta")
            plt.ylabel("Qc")
            plt.grid()
            plt.savefig(self.workspace + "/behavior/Qc_" + title + ".png")
            plt.close()
            plt.clf()

            fig, ax = plt.subplots()
            lims_x = (-1.1, 1.1)
            lims_y = (-1.1, 1.1)
            for ia in actions:
                plt.plot(yc[:, ia], yr[:, ia], ls="-", marker='v', markersize=2)
                lims_x = update_lims(lims_x, yc[:, ia])
                lims_y = update_lims(lims_y, yr[:, ia])
            if self.N_actions < 4:
                plt.legend([self.actions_str[a] for a in actions])
            plt.title(title)
            plt.xlim(*lims_x)
            plt.ylim(*lims_y)
            plt.xlabel("Qc")
            plt.ylabel("Qr")
            plt.grid()
            plt.savefig(self.workspace + "/behavior/QrQc_" + title + ".png")
            plt.close()

    def __len__(self):
        return self.sb_batch.shape[0]

    def __add__(self, other):
        return ConcatDataset([self, other])


class NetBFTQ(BaseModule):
    def __init__(self, size_state, size_beta_encoder, intra_layers, n_actions,
                 activation_type="RELU",
                 reset_type="XAVIER",
                 normalize=False,
                 beta_encoder_type="LINEAR",
                 **kwargs):
        super(NetBFTQ, self).__init__(activation_type, reset_type, normalize)
        sizes = intra_layers + [2 * n_actions]
        self.beta_encoder_type = beta_encoder_type
        self.size_state = size_state
        self.size_beta_encoder = size_beta_encoder
        self.size_action = sizes[-1] / 2
        intra_layers = []
        if size_beta_encoder > 1:
            if self.beta_encoder_type == "LINEAR":
                self.beta_encoder = torch.nn.Linear(1, size_beta_encoder)
            self.concat_layer = torch.nn.Linear(size_state + size_beta_encoder, sizes[0])
        else:
            module = torch.nn.Linear(size_state + size_beta_encoder, sizes[0])
            intra_layers.append(module)
        for i in range(0, len(sizes) - 2):
            module = torch.nn.Linear(sizes[i], sizes[i + 1])
            intra_layers.append(module)
        self.linears = nn.ModuleList(intra_layers)
        self.predict = torch.nn.Linear(sizes[-2], sizes[-1])

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std

        if self.size_beta_encoder > 1:
            beta = x[:, :, -1]
            if self.beta_encoder_type == "REPEAT":
                beta = beta.repeat(1, self.size_beta_encoder)
            elif self.beta_encoder_type == "LINEAR":
                beta = self.beta_encoder(beta)
            else:
                raise "Unknown encoder type : {}".format(self.beta_encoder_type)
            state = x[:, :, 0:-1][:, 0]
            x = torch.cat((state, beta), dim=1)
            x = self.concat_layer(x)
        elif self.size_beta_encoder == 1:
            pass
        else:
            x = x[:, :, 0:-1]

        for i, layer in enumerate(self.linears):
            x = self.activation(layer(x))
        x = self.predict(x)

        return x.view(x.size(0), -1)
