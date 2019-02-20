# -*- coding: utf-8 -*-
import gc
from multiprocessing import Pool

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
from ncarrara.utils.math_utils import update_lims, near_split
from ncarrara.utils.miscelanous import pretty_format_list
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


def f(params):
    Qsb_, action_mask, betas_for_discretisation, path, hull_options = params
    hull, _ = compute_interest_points_NN_Qsb(
        Qsb=Qsb_,
        action_mask=action_mask,
        betas=betas_for_discretisation,
        disp=False,
        path=path,
    hull_options = hull_options)

    return hull


def compute_interest_points_NN_Qsb(Qsb, action_mask, betas, disp=False, path="tmp", id="default",
                                   hull_options=None):
    with torch.no_grad():
        if not type(action_mask) == type(np.zeros(1)):
            action_mask = np.asarray(action_mask)
        N_OK_actions = int(len(action_mask) - np.sum(action_mask))

        dtype = [('Qc', 'f4'), ('Qr', 'f4'), ('beta', 'f4'), ('action', 'i4')]

        if path:
            path = path + "/interest_points/"
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
        for ibeta, beta in enumerate(betas):
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
        if hull_options is not None and hull_options["decimals"] is not None:
            points = np.round(np.array(points), decimals=hull_options["decimals"])
        else:
            points = np.array(points)

        betas = np.array(betas)
        Qs = np.array(Qs)

        # on remove les duplications
        if hull_options is not None and hull_options["remove_duplicated_points"]:
            points, indices = np.unique(points,axis=0, return_index=True)
            betas = betas[indices]
            Qs = Qs[indices]

        if disp:
            plt.rcParams["figure.figsize"] = (5, 5)
            plt.plot(all_points[:, 0], all_points[:, 1], 'o', markersize=7, color="blue", alpha=0.1)
            plt.plot(points[:, 0], points[:, 1], 'o', markersize=3, color="red")
            plt.grid()

        if len(points) < 3:
            colinearity = True
        else:
            try:
                if  hull_options is not None and  hull_options["qhull_options"] is not None:
                    hull = ConvexHull(points, options=hull_options["qhull_options"])
                else:
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
        return rez, colinearity  # betas, points, idxs_interest_points, Qs, colinearity


def compute_interest_points_NN(s, Q, action_mask, betas, device, hull_options, disp=False, path=None, id="default"):
    ss = s.repeat((len(betas), 1, 1))
    bb = torch.from_numpy(betas).float().unsqueeze(1).unsqueeze(1).to(device=device)
    sb = torch.cat((ss, bb), dim=2)
    Qsb = Q(sb)
    return compute_interest_points_NN_Qsb(Qsb, action_mask, betas, disp=disp, path=path, id=id,
                                          hull_options=hull_options)


def convex_hull(s, action_mask, Q, disp, betas, device, hull_options, path=None, id="default"):
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
        id=id,
        hull_options=hull_options)
    return hull


def optimal_pia_pib_parralle(args):
    return optimal_pia_pib(*args)


def optimal_pia_pib(beta, hull, statistic):
    with torch.no_grad():
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
                    res = OptimalPolicy(action_inf, 0, 1., 0., beta_inf,
                                        0.)  # action_inf = action_sup, beta_inf=beta_sup
        # statistic["status"] = status
        return res, status


class PytorchBudgetedFittedQ:
    ALL_BATCH = "ALL_BATCH"
    ADAPTATIVE = "ADAPTATIVE"
    DONT_COMPUTE = -1

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
                 beta_range=[],
                 # disp_states=[],
                 # disp_next_states=[],
                 workspace="tmp",
                 print_q_function=False,
                 state_to_unique_str=lambda s: str(s),
                 action_to_unique_str=lambda a: str(a),
                 use_data_loader=False,
                 use_data_parallel=False,
                 use_extra_gpu_memory_threshold=0,
                 maximum_number_of_gpu=1,
                 cpu_processes=None,
                 env=None,
                 hull_options=None

                 ):
        self.hull_options=None
        self.env = env
        self.cpu_processes = cpu_processes

        self.memory_tracking = []
        # self.track_memory("init")
        self.state_to_unique_str = state_to_unique_str
        self.action_to_unique_str = action_to_unique_str
        self.device = device
        self.print_q_function = print_q_function
        self.weights_losses = weights_losses
        self.beta_range = beta_range
        self.stop_loss_value = nn_loss_stop_condition
        self.BATCH_SIZE_EXPERIENCE_REPLAY = batch_size_experience_replay
        self.DELTA = delta_stop
        self.disp_states = []
        self.disp_next_states = []

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
            raise Exception("""Not really working. Must debug model saving and loading.""")
            logger.info("Main device : {}", self.device)
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
        self.max_print_q = 1.1
        self.reset()
        self.track_memory("__init__ (end)")

    def _construct_batch(self, transitions):
        transitions = copy.deepcopy(transitions)
        import random
        random.shuffle(transitions)

        self.track_memory("_construct_batch")
        self.info("[_construct_batch] constructing batch ...")
        memory = Memory(class_transition=TransitionBFTQ)
        if logger.getEffectiveLevel() <= logging.DEBUG:
            self.do_dynamic_disp_state = not self.disp_states
            self.do_dynamic_disp_nextstate = not self.disp_next_states
            self.display_id_state = []
            self.display_id_next_state = []
        it = 0
        last_hull_id = 0
        hull_ids = {}
        similar_next_state = []
        mask_unique_hull_ns = []
        mask_not_terminal_ns = []
        nb_terminal_ns = 0
        idx_in_batch = 0

        state_idx_in_batch_for_debug = 0
        debug_keys = {}

        for idx_transition, t in enumerate(transitions):

            debugstate_key = self.state_to_unique_str(t.state)
            if debugstate_key not in debug_keys:
                debug_keys[debugstate_key] = state_idx_in_batch_for_debug

            if it % np.ceil(len(transitions) / 10) == 0: self.info(
                "[_construct_batch] {} transitions proceeded".format(it))

            if t.next_state is not None:
                next_state = torch.tensor([[t.next_state]], dtype=torch.float)
                hull_key = self.state_to_unique_str(t.next_state)
                if hull_key in hull_ids:
                    hull_id = hull_ids[hull_key]
                    similar_next_state.append([hull_id, idx_in_batch])
                else:
                    hull_id = last_hull_id
                    mask_unique_hull_ns.append(idx_in_batch)
                    hull_ids[hull_key] = hull_id
                    last_hull_id += 1
            else:
                next_state = torch.tensor([[[np.nan] * self.size_state]], dtype=torch.float)
                hull_id = PytorchBudgetedFittedQ.DONT_COMPUTE

            hull_id = torch.tensor([[[hull_id]]], dtype=torch.long)
            action = torch.tensor([[t.action]], dtype=torch.long)
            reward = torch.tensor([t.reward], dtype=torch.float)
            constraint = torch.tensor([t.constraint], dtype=torch.float)
            state = torch.tensor([[t.state]], dtype=torch.float)
            if len(self.betas_for_duplication) > 0:
                for beta in self.betas_for_duplication:
                    beta = torch.tensor([[[beta]]], dtype=torch.float)
                    memory.push(state, action, reward, next_state, constraint, beta, hull_id)
                    if t.next_state is not None:
                        mask_not_terminal_ns.append(idx_in_batch)
                    else:
                        nb_terminal_ns += 1
                    idx_in_batch += 1
                    state_idx_in_batch_for_debug += 1

            else:
                beta = torch.tensor([[[t.beta]]], dtype=torch.float)
                memory.push(state, action, reward, next_state, constraint, beta, hull_id)
                if t.next_state is not None:
                    mask_not_terminal_ns.append(idx_in_batch)
                else:
                    nb_terminal_ns += 1
                idx_in_batch += 1
                state_idx_in_batch_for_debug += 1

            it += 1

        mask_unique_hull_ns = np.array(mask_unique_hull_ns)
        mask_not_terminal_ns = np.array(mask_not_terminal_ns)

        batch = memory.memory
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

        if len(transitions) != (len(mask_unique_hull_ns) +
                                nb_terminal_ns / max(1, len(self.betas_for_duplication)) +
                                len(similar_next_state)):
            raise Exception("""Something went wrong
            len(transitions={}) != (len(mask_unique_hull_ns={}) + nb_terminal_ns={} + len(similar_next_state={}))

            {} != {}
            """.format(len(transitions), len(mask_unique_hull_ns), nb_terminal_ns, len(similar_next_state),
                       len(transitions), len(mask_unique_hull_ns) + nb_terminal_ns + len(similar_next_state)))

        self.info("-----------------------------------------------------")
        self.info("--------------- BATCH INFORMATIONS ------------------")
        self.info("-----------------------------------------------------")
        self.info("#transitions : {}".format(len(memory)))
        self.info("#hull (=#unique_next_states) : {}".format(len(mask_unique_hull_ns)))
        self.info("#terminal next_states : {}".format(nb_terminal_ns))
        self.info("#similar next_states (excluding terminal states) : {}".format(len(similar_next_state)))
        self.info("sum of constraint : {}".format(constraint_batch.sum()))
        self.info("#reward >= 1 : {}".format(reward_batch[reward_batch >= 1.].sum()))
        self.info("-----------------------------------------------------")
        self.info("-----------------------------------------------------")
        self.track_memory("_construct_batch (end)")
        self.info("[_construct_batch] constructing batch ... end {}")

        if logger.getEffectiveLevel() <= logging.DEBUG:
            if self.do_dynamic_disp_state:
                self.disp_states = state_batch[list(debug_keys.values())]
                self.display_id_state = list(debug_keys.values())
            if self.do_dynamic_disp_nextstate:
                self.disp_next_states = next_state_batch[mask_unique_hull_ns]
                self.display_id_next_state = mask_unique_hull_ns
            if len(self.disp_states) > 10:
                self.disp_states = self.disp_states[:10]
                self.display_id_state = self.display_id_state[:10]
            if len(self.disp_next_states) > 10:
                self.disp_next_states = self.disp_next_states[:10]
                self.display_id_next_state = self.display_id_next_state[:10]
            self.info("Display on these states {}".format(pretty_format_list(self.display_id_state)))
            self.info("Display on these next states {}".format(pretty_format_list(self.display_id_next_state)))

        return state_beta_batch, state_batch, action_batch, reward_batch, \
               constraint_batch, next_state_batch, hull_id_batch, beta_batch, mask_unique_hull_ns, mask_not_terminal_ns

    def getsizeof(self, a, name):
        self.info("size {} : [{}{}{}]".format(name, Color.BOLD, getsizeof(a), Color.END))

    def fit(self, transitions):

        self.track_memory("fit")

        self._id_ftq_epoch = 0
        self.info("[fit] reseting network ...")
        self.reset_network()

        sb_batch, s_batch, a_batch, r_batch, c_batch, ns_batch, h_batch, b_batch, mask_unique_hull_ns, mask_not_terminal_ns = \
            self._construct_batch(transitions)

        self.delta = np.inf
        while self._id_ftq_epoch < self._MAX_FTQ_EPOCH and self.delta > self.DELTA:
            self.track_memory("e={}".format(self._id_ftq_epoch))
            # cpuStats()
            # memReport()
            self.info("-------------------------")
            _ = self._ftq_epoch(sb_batch, a_batch, r_batch, c_batch, ns_batch, h_batch, b_batch, mask_unique_hull_ns,
                                mask_not_terminal_ns)
            self.info("delta={}".format(self.delta))
            if logger.getEffectiveLevel() <= logging.DEBUG:
                self.info("Printing some debug graphics ...")
                for i_s, state in enumerate(self.disp_next_states):
                    if state is not None:
                        str_state = "next_state={}_epoch={:03}".format(i_s, self._id_ftq_epoch)
                        self.draw_Qr_and_Qc(state.cpu().numpy(), self._policy_network, str_state)

                        _ = convex_hull(s=state,
                                        Q=self._policy_network,
                                        action_mask=np.zeros(self.N_actions),
                                        id=str_state, disp=True,
                                        betas=self.betas_for_discretisation,
                                        device=self.device,
                                        path=self.workspace,
                                        hull_options=self.hull_options)
                for i_s, state in enumerate(self.disp_states):
                    if state is not None:
                        str_state = "state={}_epoch={:03}_".format(i_s, self._id_ftq_epoch)
                        self.draw_Qr_and_Qc(state.cpu().numpy(), self._policy_network, str_state)

                        _ = convex_hull(s=state,
                                        Q=self._policy_network,
                                        action_mask=np.zeros(self.N_actions),
                                        id=str_state, disp=True,
                                        betas=self.betas_for_discretisation,
                                        device=self.device,
                                        path=self.workspace,
                                        hull_options=self.hull_options)
                self.info("Printing some debug graphics ... (end)")
            self._id_ftq_epoch += 1

        return self._policy_network

    def _ftq_epoch(self, sb_batch, a_batch, r_batch, c_batch, ns_batch, h_batch, b_batch, mask_unique_hull_ns,
                   mask_not_terminal_ns):
        self.track_memory("ftq_epoch")
        self.info("[_ftq_epoch] start ...")
        with torch.no_grad():
            if self._id_ftq_epoch > 0:
                ns_r, ns_c = self.compute_next_values(ns_batch, h_batch, b_batch, self._policy_network,
                                                      mask_unique_hull_ns, mask_not_terminal_ns)

            else:
                ns_r = torch.zeros(self.size_batch, device=self.device)
                ns_c = torch.zeros(self.size_batch, device=self.device)

            label_r = r_batch + (self._GAMMA * ns_r)
            label_c = c_batch + (self._GAMMA_C * ns_c)

            if self.beta_range:
                self.info("Clamp target constraints")
                label_c = torch.clamp(label_c, min=self.beta_range[0], max=self.beta_range[1])

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

                from ncarrara.utils_rl.environments.gridworld.envgridworld import EnvGridWorld
                if isinstance(self.env, EnvGridWorld):
                    def pi(state, beta):
                        import torch
                        from ncarrara.budgeted_rl.bftq.pytorch_budgeted_fittedq import convex_hull, \
                            optimal_pia_pib
                        with torch.no_grad():
                            hull = convex_hull(s=torch.tensor([state], device=self.device, dtype=torch.float32),
                                               Q=self._policy_network,
                                               action_mask=np.zeros(self.env.action_space.n),
                                               id="run_" + str(state), disp=False,
                                               betas=self.betas_for_discretisation,
                                               device=self.device)
                            opt, _ = optimal_pia_pib(beta=beta, hull=hull, statistic={})
                        return opt

                    def qr(state, a, beta):
                        import torch
                        s = torch.tensor([[state]], device=self.device)
                        b = torch.tensor([[[beta]]], device=self.device)
                        sb = torch.cat((s, b), dim=2)
                        return self._policy_network(sb).squeeze()[a]

                    def qc(state, a, beta):
                        import torch
                        s = torch.tensor([[state]], device=self.device)
                        b = torch.tensor([[[beta]]], device=self.device)
                        sb = torch.cat((s, b), dim=2)
                        return self._policy_network(sb).squeeze()[self.env.action_space.n + a]

                    from ncarrara.utils_rl.environments.gridworld.world import World
                    w = World(self.env, self.betas_for_discretisation)
                    w.draw_frame()
                    w.draw_lattice()
                    w.draw_cases()
                    w.draw_policy_bftq(pi, qr, qc, self.betas_for_discretisation)
                    w.save(self.workspace + "/bftq_on_2dworld_e_={}".format(self._id_ftq_epoch))

        self.empty_cache()
        self.info("[_ftq_epoch] ... end")
        self.track_memory("ftq_epoch (end)")
        return losses

    def compute_next_values(self, ns_batch, h_batch, b_batch, Q, where_unique_hull_ns, where_not_terminal_ns):
        next_state_rewards = torch.zeros(self.size_batch, device=self.device)
        next_state_constraints = torch.zeros(self.size_batch, device=self.device)

        if len(where_not_terminal_ns) > 0:
            with torch.no_grad():
                ################################################
                # computing all unique hulls (no terminal next_state and no repetition among next state)
                ################################################
                self.track_memory("compute hulls")
                self.info("computing hulls ...")

                # gather all s,beta to compute in one foward pass (may increase the pic of memory used temporary)
                # but essential for multiprocessing hull computing bellow

                ns_batch_unique = ns_batch[where_unique_hull_ns]

                self.info("There are {} hulls to compute !".format(len(ns_batch_unique)))

                """ 
                sb = "duplicating each next states with each beta"
                if we have 2 states s0 and s1 and 2 betas 0 and 1. What do we want for sb is :
                (s0,0)
                (s0,1)
                (s1,0)
                (s1,1)
                """
                ss = ns_batch_unique \
                    .squeeze() \
                    .repeat((1, len(self.betas_for_discretisation))) \
                    .reshape(-1) \
                    .reshape((len(ns_batch_unique) * len(self.betas_for_discretisation), self.size_state))
                bb = torch.from_numpy(self.betas_for_discretisation).float().unsqueeze(1).to(device=self.device)
                bb = bb.repeat((len(ns_batch_unique), 1))
                sb = torch.cat((ss, bb), dim=1)
                sb = sb.unsqueeze(1)

                self.track_memory("Qsb (compute_hull) ")
                # self.info("Forward pass of couple (s',beta). Size of the batch : {}." +
                #           "It should be equals to #hulls({}) x #beta_for_discretisation({})  : {}"
                #           .format(len(sb), len(ns_batch_unique), len(self.betas_for_discretisation),
                #                   len(self.betas_for_discretisation) * len(ns_batch_unique)))
                num_bins = 5
                batch_sizes = near_split(x=len(sb), num_bins=num_bins)
                y = []
                self.info("Splitting x in minibatch to avoid out of memory")
                self.info("Batch sizes : {}".format(batch_sizes))
                offset = 0
                for i in range(num_bins):
                    self.info("mini batch {}".format(i))
                    self.track_memory("mini_batch={}".format(i))
                    mini_batch = sb[offset:offset + batch_sizes[i]]
                    offset += batch_sizes[i]
                    y.append(Q(mini_batch))
                    torch.cuda.empty_cache()
                Qsb = torch.cat(y)
                Qsb = Qsb.detach().cpu().numpy()
                self.track_memory("Qsb (compute_hull) (end)")

                args_for_ns_batch_unique = [
                    (
                        Qsb[(i_ns_unique * len(self.betas_for_discretisation)):
                            (i_ns_unique + 1) * len(self.betas_for_discretisation)],
                        np.zeros(self.N_actions),
                        self.betas_for_discretisation,
                        self.workspace,
                        self.hull_options
                    )
                    for i_ns_unique in range(len(ns_batch_unique))
                ]

                if self.cpu_processes == 1:
                    hulls_for_ns_batch_unique = []
                    for i_params, params in enumerate(args_for_ns_batch_unique):
                        if i_params % max(1, len(args_for_ns_batch_unique) // 10) == 0:
                            self.info("{} hulls processed (sequentially)".format(i_params))
                        hulls_for_ns_batch_unique.append(f(params))
                else:
                    self.info("Using multiprocessing")
                    with Pool(self.cpu_processes) as p:
                        # p.map return ordered fashion, so we're cool
                        hulls_for_ns_batch_unique = p.map(f, args_for_ns_batch_unique)

                self.info("#next_states : {}".format(len(ns_batch)))
                self.info("#non terminal next_states : {}".format(len(where_not_terminal_ns)))
                self.info("#hulls actually computed : {}".format(len(hulls_for_ns_batch_unique)))
                self.info("computing hulls [DONE] ")
                self.empty_cache()
                self.track_memory("compute hulls (end)")

                #####################################
                # for each next_state
                # computing optimal distribution among 2 actions,
                # with respectively 2 budget
                # given the hull of the next_state and the beta (in current state).
                #####################################

                self.track_memory("compute_opts")
                self.info("computing ops ... ")

                ##############################################################
                # we build couples (s',beta\top) and (s',beta\bot)
                # in order to compute Q(s')
                ##############################################################
                next_state_beta_not_terminal = torch.zeros((len(where_not_terminal_ns) * 2, 1, self.size_state + 1),
                                                           device=self.device)
                ns_batch_not_terminal = ns_batch[where_not_terminal_ns]
                h_batch_not_terminal = h_batch[where_not_terminal_ns]
                b_bath_not_terminal = b_batch[where_not_terminal_ns]

                args = [(beta.detach().item(), hulls_for_ns_batch_unique[hull_id], {})
                        for hull_id, beta in zip(h_batch_not_terminal, b_bath_not_terminal)]

                if self.cpu_processes == 1:
                    opts_and_statuses = []
                    for i_arg, arg in enumerate(args):
                        if i_arg % (max(1, len(args) // 10)) == 0:
                            self.info("{} opts proccessed (sequentially)".format(i_arg))
                        opts_and_statuses.append(optimal_pia_pib_parralle(arg))
                else:
                    self.info("computing optimal_pia_pib in parralle ...")
                    with Pool(self.cpu_processes) as p:
                        opts_and_statuses = p.map(optimal_pia_pib_parralle, args)
                    self.info("computing optimal_pia_pib in parralle ... done")

                self.info("computing opts ... end")
                self.track_memory("compute_opts (end)")
                status = {"regular": 0, "not_solvable": 0, "too_much_budget": 0, "exact": 0}

                for i_ns_nt, (ns_not_terminal, opt_and_status) in enumerate(
                        zip(ns_batch_not_terminal, opts_and_statuses)):
                    opt, stat = opt_and_status
                    status[stat] += 1
                    ns_beta_moins = torch.cat((ns_not_terminal, torch.tensor([[opt.budget_inf]], device=self.device)),
                                              dim=1)
                    ns_beta_plus = torch.cat((ns_not_terminal, torch.tensor([[opt.budget_sup]], device=self.device)),
                                             dim=1)
                    next_state_beta_not_terminal[i_ns_nt * 2 + 0][0] = ns_beta_moins
                    next_state_beta_not_terminal[i_ns_nt * 2 + 1][0] = ns_beta_plus

                ##############################################################
                # Forwarding to compute the Q function in s' #################
                ##############################################################

                self.info("Q next")
                self.track_memory("Q_next")
                Q_next_state_not_terminal = self._policy_network(next_state_beta_not_terminal)
                Q_next_state_not_terminal = Q_next_state_not_terminal.detach()
                self.track_memory("Q_next (end)")
                self.info("Q next end")
                self.empty_cache()

                ###########################################
                ############  bootstraping   ##############
                ###########################################

                self.info("computing next values ...")
                self.track_memory("compute_next_values")

                warning_qc_negatif = 0.
                offset_qc_negatif = 0.
                warning_qc__negatif = 0.
                offset_qc__negatif = 0.
                next_state_c_neg = 0.
                offset_next_state_c_neg = 0.

                next_state_rewards_not_terminal = torch.zeros(len(where_not_terminal_ns), device=self.device)
                next_state_constraints_not_terminal = torch.zeros(len(where_not_terminal_ns), device=self.device)

                for i_ns_nt in range(len(where_not_terminal_ns)):
                    opt = opts_and_statuses[i_ns_nt][0]
                    qr_ = Q_next_state_not_terminal[i_ns_nt * 2][opt.id_action_inf]
                    qr = Q_next_state_not_terminal[i_ns_nt * 2 + 1][opt.id_action_sup]
                    qc_ = Q_next_state_not_terminal[i_ns_nt * 2][self.N_actions + opt.id_action_inf]
                    qc = Q_next_state_not_terminal[i_ns_nt * 2 + 1][self.N_actions + opt.id_action_sup]

                    if qc < 0.:
                        warning_qc_negatif += 1.
                        offset_qc_negatif = qc

                    if qc_ < 0.:
                        warning_qc__negatif += 1.
                        offset_qc__negatif = qc_

                    next_state_rewards_not_terminal[i_ns_nt] = opt.proba_inf * qr_ + opt.proba_sup * qr
                    next_state_constraints_not_terminal[i_ns_nt] = opt.proba_inf * qc_ + opt.proba_sup * qc

                    if next_state_constraints_not_terminal[i_ns_nt] < 0:
                        next_state_c_neg += 1.
                        offset_next_state_c_neg = next_state_constraints_not_terminal[i_ns_nt]

                next_state_rewards[where_not_terminal_ns] = next_state_rewards_not_terminal
                next_state_constraints[where_not_terminal_ns] = next_state_constraints_not_terminal

                if logger.getEffectiveLevel() <= logging.DEBUG:
                    self.info("printing some graphs in next_values ...")
                    self.info("\n[compute_next_values] Q(s') sur le batch")
                    create_Q_histograms("Qr(s')_e={}".format(self._id_ftq_epoch),
                                        values=next_state_rewards.cpu().numpy().flatten(),
                                        path=self.workspace + "/histogram",
                                        labels=["next value"])
                    create_Q_histograms("Qc(s')_e={}".format(self._id_ftq_epoch),
                                        values=next_state_constraints.cpu().numpy().flatten(),
                                        path=self.workspace + "/histogram",
                                        labels=["next value"])
                    self.info("printing some graphs in next_values ... done")

                mean_qc_neg = 0 if warning_qc_negatif == 0 else offset_qc_negatif / warning_qc_negatif
                mean_qc__neg = 0 if warning_qc__negatif == 0 else offset_qc__negatif / warning_qc__negatif
                mean_ns_neg = 0 if next_state_c_neg == 0 else offset_next_state_c_neg / next_state_c_neg
                self.info("qc < 0 percentage {:.2f}% with a mean offset of {:.4f}".format(
                    warning_qc_negatif / len(where_not_terminal_ns) * 100., mean_qc_neg))
                self.info("qc_ < 0 percentage {:.2f}% with a mean offset of {:.4f}".format(
                    warning_qc__negatif / len(where_not_terminal_ns) * 100., mean_qc__neg))
                self.info("next_state_constraints < 0 percentage {:.2f}% with a mean offset of {:.4f}".format(
                    next_state_c_neg / len(where_not_terminal_ns) * 100., mean_ns_neg))
                self.info("compute next values ... end")
                self.empty_cache()
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
        if False and self.use_data_loader:
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
        state_action_constraints = output.gather(1, a_batch + self.N_actions)
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

    def draw_Qr_and_Qc(self, s, Q, id):
        with torch.no_grad():
            plt.rcParams["figure.figsize"] = (5, 5)
            plt.clf()
            actions = range(self.N_actions)  # [2
            if not os.path.exists(self.workspace + "/behavior/"):
                os.makedirs(self.workspace + "/behavior/")
            betas = self.betas_for_discretisation  # np.linspace(0, self.beta_max, 100)
            title = id
            yr = np.zeros((len(betas), self.N_actions))
            yc = np.zeros((len(betas), self.N_actions))
            for idx, beta in enumerate(betas):
                qrqc = Q(torch.tensor([[np.append(s, beta)]], device=self.device).float()).cpu().detach().numpy()
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

    def save_policy(self, policy_path=None):
        if policy_path is None:
            policy_path = self.workspace + "/policy.pt"
        self.info("saving bftq policy at {}".format(policy_path))
        torch.save(self._policy_network, policy_path)
        return policy_path

    def _is_terminal_state(self, state):
        isnan = torch.sum(torch.isnan(state)) == self.size_state
        return isnan

    def empty_cache(self):
        memoire_before = get_memory_for_pid(os.getpid())
        torch.cuda.empty_cache()
        memoire_after = get_memory_for_pid(os.getpid())
        self.info("empty cache {} -> {}".format(self.format_memory(memoire_before), self.format_memory(memoire_after)))

    def reset_network(self):
        if self.use_data_parallel:
            self._policy_network.module.reset()
        else:
            self._policy_network.reset()

    def reset(self, reset_weight=True):
        torch.cuda.empty_cache()
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
            logger.info("[e={:02}]{} {}".format(self._id_ftq_epoch, self.format_memory(memoire), message))
        else:
            logger.info("{} {} ".format(self.format_memory(memoire), message))


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
