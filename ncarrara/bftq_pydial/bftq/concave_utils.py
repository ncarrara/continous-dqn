#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
import matplotlib.pyplot as plt
from numpy.random import random_sample
import sys
import os


# PENSER A FAIRE CA
# next_reward = self.compute_max_value(actions, sp, q) if not info["state_is_absorbing"] else 0.

def compute_interest_points(fonctions, N, min_x=0.0, max_x=1.0, disp=False, path=None):
    step = max_x / float(N)
    colinearity = False
    N_fonctions = len(fonctions)
    points = np.zeros((N + 1, 2))
    all_points = np.zeros((N_fonctions * (N + 1), 2))
    # indexes = []
    idxs_corresponding_function = []
    x = min_x
    j = 0
    for k in range(0, N + 1):
        indexe_max_f = None
        max = -np.inf
        for index_f in range(0, len(fonctions)):
            value = fonctions[index_f](x)
            all_points[j] = [x, value]
            if value > max:
                indexe_max_f = index_f
                max = value
            j += 1
        # indexes.append(indexe_max_f)
        idxs_corresponding_function.append(indexe_max_f)
        points[k] = [x, max]  # max_a
        x += step
    if disp:
        plt.plot(all_points[:, 0], all_points[:, 1], 'o')
        plt.plot(points[:, 0], points[:, 1], '-')
        # plt.show()
    try:
        hull = ConvexHull(points)
    except QhullError:
        # ca veut dire colinéaire, donc une fonction meilleur que les autres et constants (ou toutes fonctions equivalents)
        interest_idxs = range(0, N + 1)
        colinearity = True
        if disp:
            plt.plot(points[interest_idxs][:, 0], points[interest_idxs][:, 1], 'x')
            plt.plot(points[interest_idxs, 0], points[interest_idxs, 1], 'r--', lw=2)
            plt.show()
        return points, interest_idxs, idxs_corresponding_function, colinearity
    # On ne garde que le dessus,
    # comme c'est counterclockwise
    # on peut faire ça :
    k = 0
    stop = False
    min_y = None
    # on cherche le point qui est en abcisse x_min
    while not stop:
        idx_vertex = hull.vertices[k]
        x, y = points[idx_vertex]
        if np.abs(x - min_x) < 0.0001:
            stop = True
            min_y = y
        else:
            k = (k + 1) % len(hull.vertices)
    interest_idxs_corresponding_f = []
    interest_idxs = []
    stop = False
    # on va itera a l'envers jusq'a etre inférieur
    y_previous = -np.inf
    while not stop:
        idx_vertex = hull.vertices[k]
        x, y = points[idx_vertex]
        if y_previous < y:
            interest_idxs.append(idx_vertex)
            interest_idxs_corresponding_f.append(idxs_corresponding_function[idx_vertex])
        if np.abs(x - max_x) < 0.0001 or y <= y_previous:
            stop = True
        else:
            k = (k - 1) % len(hull.vertices)
            y_previous = y
    if disp:
        # plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2, color="green")
        plt.plot(points[interest_idxs][:, 0], points[interest_idxs][:, 1], 'x')
        plt.plot(points[interest_idxs, 0], points[interest_idxs, 1], 'r--', lw=2)

        plt.show()
    return points, interest_idxs, interest_idxs_corresponding_f, colinearity


def main(N=100):
    Q = lambda x: random_sample()

    compute_interest_points([Q], N=N, min_x=0.0, max_x=1.0, disp=True)

    Q0 = lambda x: 1 + 5 * x

    def Q1(x):
        return 5 + np.log(0.1 + x)

    compute_interest_points([Q0, Q1], N=N, min_x=0.0, max_x=1.0, disp=True)

    Q0 = lambda x: 2 + np.log(0.1 + 100 * x)

    def Q1(x):
        return 5 + np.log(0.1 + x)

    compute_interest_points([Q0, Q1], N=N, min_x=0.0, max_x=1.0, disp=True)

    Q0 = lambda beta: 10

    def Q1(beta):
        if beta < 0.99:
            return 0
        else:
            return 100

    compute_interest_points([Q0, Q1], N=N, min_x=0.0, max_x=2.0, disp=True)

    Q0 = lambda x: 0
    Q1 = lambda x: 0.3
    Q2 = lambda x: 1.0
    Q3 = lambda x: 0 if x < 0.99 else 0.1

    compute_interest_points([Q0, Q1, Q2, Q3], N=N, min_x=0.0, max_x=1.0, disp=True)


if __name__ == "__main__":
    main(N=1000)


def compute_interest_points_from_Qc_Qr_betas(Q_as, betas, disp=False, path=None, id="default"):
    # print betas
    dtype = [('Qc', 'f4'), ('Qr', 'f4'), ('beta', 'f4'), ('action', 'i4')]

    path = path + "/interest_points/"
    # print path
    colinearity = False
    test = False
    if disp or test:
        if not os.path.exists(path):
            os.makedirs(path)

    all_points = np.zeros((len(Q_as) * len(betas), 2))
    all_betas = np.zeros((len(Q_as) * len(betas),))
    all_Qs = np.zeros((len(Q_as) * len(betas),), dtype=int)
    max_Qr = -np.inf
    Qc_for_max_Qr = None
    l = 0
    x = np.zeros((len(Q_as), len(betas)))
    y = np.zeros((len(Q_as), len(betas)))
    i_beta = 0
    for beta in betas:
        for i_a in range(0, len(Q_as)):
            # print "hello", Q_as[i_a](beta)
            Qc, Qr = Q_as[i_a](beta)
            x[i_a][i_beta] = Qc
            y[i_a][i_beta] = Qr
            if Qr > max_Qr:
                max_Qr = Qr
                Qc_for_max_Qr = Qc
            all_points[l] = (Qc, Qr)
            all_Qs[l] = i_a
            all_betas[l] = beta
            l += 1
        i_beta += 1
    if disp or test:
        for i_a in range(0, len(Q_as)):
            # print "-------------"
            # print x[i_a]
            # print y[i_a]
            plt.plot(x[i_a], y[i_a], linewidth=8, alpha=0.5)
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
    if disp or test:
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

    if disp or test:
        plt.title(
            "compute_interest_points_from_Qc_Qr_colinear={}".format(colinearity))
        plt.plot(points[idxs_interest_points, 0], points[idxs_interest_points, 1], 'r--', lw=1, color="red")
        plt.plot(points[idxs_interest_points][:, 0], points[idxs_interest_points][:, 1], 'x', markersize=20,
                 color="tab:pink")

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


import torch


def compute_interest_points_NN(s, Q, action_mask, betas, device, disp=False, path=None, id="default"):
    if not type(action_mask) == type(np.zeros(1)):
        action_mask = np.asarray(action_mask)
    # print betas
    N_OK_actions = int(len(action_mask) - np.sum(action_mask))

    dtype = [('Qc', 'f4'), ('Qr', 'f4'), ('beta', 'f4'), ('action', 'i4')]

    path = path + "/interest_points/"
    # print path
    colinearity = False
    test = False
    if disp or test:
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
    for beta in betas:
        with torch.no_grad():
            b_ = torch.tensor([[beta]], device=device, dtype=torch.float32)
            sb = torch.cat((s.float(), b_), 1)
            sb = sb.unsqueeze(0)
            QQ = Q(sb)[0]
            QQ = QQ.cpu().detach().numpy()
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

    if disp or test:
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
    if disp or test:
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

    if disp or test:
        plt.title(
            "interest_points_colinear={}".format(colinearity))
        # plt.xlim(-0.1, 1.1)
        # plt.ylim(-0.1, 1.1)
        plt.plot(points[idxs_interest_points, 0], points[idxs_interest_points, 1], 'r--', lw=1, color="red")
        plt.plot(points[idxs_interest_points][:, 0], points[idxs_interest_points][:, 1], 'x', markersize=15,
                 color="tab:pink")

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
