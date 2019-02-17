# -*- coding: utf-8 -*-
from ncarrara.utils_rl.environments.gridworld.envgridworld import EnvGridWorld
import random
import numpy as np

np.set_printoptions(precision=3, suppress=True)


def generate(
        N_models=5,
        dim=(10, 10),
        std=(0.5, 0.5),
        trajectory_max_size=100,
        nb_holes=25,
        penalty_on_move=0.,
        hole_penalty=0.,
        goal_reward=1.,
        noise_type="gaussian_bis"):  # int(random.random() * len(holes)
    dim_x, dim_y = dim
    start = (0.5, 0.5)
    nb_cases = dim_x * dim_y
    if nb_holes is None:
        nb_holes = int(0.1 * nb_cases)
    # print "nb_holes", nb_holes
    if trajectory_max_size is None:
        trajectory_max_size = int((dim_x + dim_y) + (dim_x + dim_y) / 2.)
    sources = []
    goals = [((dim_x - 1, dim_y - 1, dim_x, dim_y), goal_reward, 0, True)]
    mu, sigma = 0.5, 0.1  # mean and standard deviation
    dev = np.random.normal(mu, sigma)
    if dev > 1 or dev < 0:
        dev = random.random()
    sx = 0
    sy = 0
    admissible_trajectory = [(sx, sy)]
    # print dev

    while sx != dim_x - 1 or sy != dim_y - 1:
        ax, ay = (0, 1) if random.random() < dev else (1, 0)
        x = sx + ax
        y = sy + ay
        if not (x < 0 or x > dim_x - 0.5 or y < 0 or y > dim_y - 0.5):
            sx = x
            sy = y
            admissible_trajectory.append((sx, sy))

    # print admissible_trajectory
    holes = []
    for i in range(0, dim_x):
        for j in range(0, dim_y):
            if (i, j) not in admissible_trajectory:
                holes.append(((i, j, i + 1, j + 1), -hole_penalty, 1, True))
    # print holes
    cases = holes + goals
    union = EnvGridWorld(dim, std, cases, trajectory_max_size, True, noise_type=noise_type, id="union",
                               penalty_on_move=penalty_on_move, init_s=start)

    for l in range(0, N_models):
        h = []
        indexes = []
        nb = 0
        while nb < nb_holes:
            index = int(random.random() * len(holes))
            if index not in indexes:
                indexes.append(index)
                nb += 1
                h.append(holes[index])
        # print indexes
        cases = h + goals
        source = EnvGridWorld(dim, std, cases, trajectory_max_size, True, noise_type=noise_type,
                                    id="source_{}".format(l), penalty_on_move=penalty_on_move, init_s=start)
        sources.append(source)
    return union, sources


def generate_random_model(
        dim=(10, 10),
        std=(0.5, 0.5),
        trajectory_max_size=None,
        nb_holes=None,
        noise_type="normalized_gaussian",
        penalty_on_move=0., goal_reward=1., nb_path_to_goal=2,
        absorbing_holes=True, A=None, A_str=None
):  # int(random.random() * len(holes)
    dim_x, dim_y = dim
    nb_cases = dim_x * dim_y
    if nb_holes is None:
        nb_holes = int(0.15 * nb_cases)
    # print "nb_holes", nb_holes
    if trajectory_max_size is None:
        trajectory_max_size = int((dim_x + dim_y) + (dim_x + dim_y) / 2.)
    sources = []
    goals = [((dim_x - 1, dim_y - 1, dim_x, dim_y), goal_reward, 0,
              True)]  # ,((0, dim_y - 1, 1, dim_y), goal_reward/2., 0, True)]
    mu, sigma = 0.5, 0.1  # mean and standard deviation

    admissible_trajectory = [(0, 0)]

    for _ in range(0, nb_path_to_goal):
        dev = np.random.normal(mu, sigma)
        if dev > 1 or dev < 0:
            dev = random.random()
        sx = 0
        sy = 0
        while sx != dim_x - 1 or sy != dim_y - 1:
            ax, ay = (0, 1) if random.random() < dev else (1, 0)
            x = sx + ax
            y = sy + ay
            if not (x < 0 or x > dim_x - 0.5 or y < 0 or y > dim_y - 0.5):
                sx = x
                sy = y
                admissible_trajectory.append((sx, sy))

    holes = []

    for i in range(0, dim_x):
        for j in range(0, dim_y):
            if (i, j) not in admissible_trajectory:
                holes.append(((i, j, i + 1, j + 1), 0, 1, absorbing_holes))

    indexes = []
    nb = 0
    h = []
    k = 0
    while nb < nb_holes:
        index = int(random.random() * len(holes))
        if index not in indexes:
            indexes.append(index)
            nb += 1
            h.append(holes[index])
        if k > 1000:
            raise Exception("dont find a way")
        k += 1
    # print indexes
    cases = h + goals
    start = (.5, .5)
    if A is None:
        A = [(0., 0.), (0., 1.), (1., 0.), (0.707107, 0.707107)]
        A_str = ["X", "E", "S", "SE"]  # ,
    m = EnvGridWorld(dim, std, cases, trajectory_max_size, True, noise_type=noise_type, init_s=start, actions=A,
                           actions_str=A_str, penalty_on_move=penalty_on_move)

    emax = dim_x + dim_y + 5
    return m, emax


def generate_test_0(std=(0.25, 0.25), trajectory_max_size=None, noise_type="normalized_gaussian"):
    dim = (7, 7)
    dim_x, dim_y = dim
    if trajectory_max_size is None:
        trajectory_max_size = int((dim_x + dim_y) + (dim_x + dim_y) / 2.)
    goals = [((6, 6, 7, 7), 100, 0, True)]
    holes = [((3, 3, 4, 4), 0, 1, False), ((4, 3, 5, 4), 0, 1, False)]

    cases = holes + goals
    start = (.5, .5)
    A = [(0., 0.), (0., 1.), (0.414213, 0.414213), (1., 0.), (0., 0.5), (0.5, 0),
         (0.707106, 0.707106)]  # ,(-1.,0.),(0,-1.)]
    A_str = ["X", "E", "SE", "S", "e", "s", "se"]  # ,"<-","^"]
    m = EnvGridWorld(dim, std, cases, trajectory_max_size, True, penalty_on_move=0., noise_type=noise_type,
                           init_s=start, actions=A, actions_str=A_str)

    emax = dim_x + dim_y + 5  # voir plus avec bruit gaussian et aller retour
    return m, emax


def generate_test_1(std=(0.5, 0.5), trajectory_max_size=None, noise_type="gaussian_bis"):
    dim = (5, 5)
    dim_x, dim_y = dim
    if trajectory_max_size is None:
        trajectory_max_size = int((dim_x + dim_y) + (dim_x + dim_y) / 2.)
    goals = [((4, 4, 5, 5), 10, 0, True), ((2, 4, 3, 5), 100, 0, True), ((2, 0, 3, 1), 1, 0, True)]
    holes = [((2, 2, 3, 3), 0, 1, True), ((0, 3, 1, 5), 0, 1, True)]

    cases = holes + goals
    start = (.5, .5)
    A = [(0., 0.), (0., 1.), (1., 0.)]  # ,(-1.,0.),(0,-1.)]
    A_str = ["X", "v", ">"]  # ,"<-","^"]
    m = EnvGridWorld(dim, std, cases, trajectory_max_size, True, noise_type=noise_type, init_s=start, actions=A,
                           actions_str=A_str, penalty_on_move=0.)

    emax = dim_x + dim_y + 5  # voir plus avec bruit gaussian et aller retour
    return m, emax


def generate_test_2(std=(0.75, 0.75), trajectory_max_size=None, noise_type="gaussian_bis"):
    dim = (3, 4)
    dim_x, dim_y = dim
    if trajectory_max_size is None:
        trajectory_max_size = int((dim_x + dim_y) + (dim_x + dim_y) / 2.)
    # goals = [((1, 0, 2, 1), 0.01, 0, True), ((2, 0, 3, 1), 0.1, 0, True), ((0, 3, 1, 4), 1., 0, True)]
    goals = [((0, 3, 1, 4), 1., 0, True)]
    holes = [((1, 0, 2, 1), 0.0, 1., True), ((2, 0, 3, 1), 0., 1., True),((0, 2, 2, 3), 0, 1, True)]

    cases = holes + goals
    A = [(0., 0.), (0., 1.), (1., 0.), (-1., 0), (0., -1.)]  # ,(-1.,0.),(0,-1.)]
    A_str = ["X", "v", ">", "<", "^"]  # ,"<-","^"]
    start = np.array([.5, .5])
    m = EnvGridWorld(dim, std, cases, trajectory_max_size, True, noise_type=noise_type, init_s=start, actions=A,
                           actions_str=A_str)
    emax = dim_x + dim_y + 10  # voir plus avec bruit gaussian et aller retour
    return m, emax


def generate_test_boucle(std=(0.5, 0.5), trajectory_max_size=None, noise_type="gaussian_bis"):
    dim = (2, 3)
    dim_x, dim_y = dim
    if trajectory_max_size is None:
        trajectory_max_size = int((dim_x + dim_y) + (dim_x + dim_y) / 2.)
    goals = [((0, 2, 1, 3), 1, 0, True)]
    holes = [((0, 1, 1, 2), 0, 1, True)]

    cases = holes + goals
    A = [(0., 0.), (0., 1.), (1., 0.), (-1., 0), (0., -1.)]  # ,(-1.,0.),(0,-1.)]
    A_str = ["X", "v", "->", "<-", "^"]  # ,"<-","^"]
    start = np.array([.5, .5])
    m = EnvGridWorld(dim, std, cases, trajectory_max_size, True, noise_type=noise_type, init_s=start, actions=A,
                           actions_str=A_str)
    emax = dim_x + dim_y + 10  # voir plus avec bruit gaussian et aller retour
    return m, emax


def generate_test_death_trap():
    dim = (2, 2)
    start = np.array([.5, .5])
    A = [(0., 1.), (1., 0.)]
    A_str = ["||", "="]
    std = None
    trajectory_max_size = 1
    goals = [((0, 1, 1, 2), 10, 0, True), ((1, 1, 2, 2), 100, 0, True)]
    holes = [((1, 0, 2, 1), 0, 1, True)]
    cases = holes + goals
    m = EnvGridWorld(dim, std, cases, trajectory_max_size, True, noise_type="test_death_trap", init_s=start,
                           actions=A, actions_str=A_str)
    emax = 1
    return m, emax


def generate_3xWidth(width=3):
    trajectory_max_size = width + 1
    A_str = ["top", "right"]
    A = [(0., -1.), (1., 0.)]
    start = np.array([.5, 1.5])
    goals = []
    holes = []
    for w in range(0, width):
        goals.append(((w, 0, w + 1, 1), pow(10, w), 0, True))
        holes.append(((w, 2, w + 1, 3), 0, 1, True))
    cases = goals + holes
    dim = (width, 3)
    std = None
    m = EnvGridWorld(dim, std, cases, trajectory_max_size, True, noise_type="3xWidth", init_s=start, actions=A,
                           actions_str=A_str)
    emax = width
    return m, emax


def generate_continuous3xWidth(width=3):
    trajectory_max_size = width + 1
    A_str = ["top", "right"]
    A = [(0., -1.), (1., 0.)]
    start = np.array([.5, 1.5])
    goals = []
    holes = []
    for w in range(0, width):
        goals.append(((w, 0, w + 1, 1), pow(10, w), 0, True))
        holes.append(((w, 2, w + 1, 3), 0, 1, True))
    cases = goals + holes
    dim = (width, 3)
    std = (0.5, 0.5)
    m = EnvGridWorld(dim, std, cases, trajectory_max_size, True, noise_type="gaussian_bis", init_s=start,
                           actions=A,
                           actions_str=A_str)
    emax = np.ceil(width * 1.5)
    return m, emax


def generate_test_3(std=(0.5, 0.5), trajectory_max_size=None, noise_type="gaussian_bis"):
    dim = (3, 3)
    dim_x, dim_y = dim
    if trajectory_max_size is None:
        trajectory_max_size = int((dim_x + dim_y) + (dim_x + dim_y) / 2.)
    goals = [((2, 2, 3, 3), 1, 0, True)]
    holes = [((1, 0, 3, 2), 0, 1, True)]

    cases = holes + goals
    A = [(0., 0.), (0., 1.), (1., 0.)]  # ,(-1.,0.),(0,-1.)]
    A_str = ["X", "v", "->"]  # ,"<-","^"]
    start = np.array([.5, .5])
    m = EnvGridWorld(dim, std, cases, trajectory_max_size, True, noise_type=noise_type, init_s=start, actions=A,
                           actions_str=A_str)
    emax = dim_x + dim_y + 2  # voir plus avec bruit gaussian et aller retour
    return m, emax


def generate_test_4(std=(0.5, 0.5), trajectory_max_size=None, noise_type="gaussian_bis"):
    dim = (4, 4)
    dim_x, dim_y = dim
    if trajectory_max_size is None:
        trajectory_max_size = int((dim_x + dim_y) + (dim_x + dim_y) / 2.)
    goals = [((3, 3, 4, 4), 1, 0, True)]
    holes = [((0, 1, 1, 3), 0, 1, True), ((2, 1, 3, 2), 0, 1, True), ((1, 3, 3, 4), 0, 1, True)]

    cases = holes + goals
    A = [(0., 0.), (0., 1.), (1., 0.)]  # ,(-1.,0.),(0,-1.)]
    A_str = ["X", "v", "->"]  # ,"<-","^"]
    start = np.array([.5, .5])
    m = EnvGridWorld(dim, std, cases, trajectory_max_size, True, noise_type=noise_type, init_s=start, actions=A,
                           actions_str=A_str)
    emax = dim_x + dim_y + 2  # voir plus avec bruit gaussian et aller retour
    return m, emax


def generate_test_5(std=(0.5, 0.5), trajectory_max_size=None, noise_type="gaussian_bis"):
    dim = (4, 4)
    dim_x, dim_y = dim
    if trajectory_max_size is None:
        trajectory_max_size = int((dim_x + dim_y) + (dim_x + dim_y) / 2.)
    goals = [((3, 3, 4, 4), 1, 0, True)]
    holes = [((2, 0, 4, 1), 0, 1, True),
             ((1, 1, 2, 2), 0, 1, True),
             ((2, 2, 3, 3), 0, 1, True),
             ((3, 1, 4, 2), 0, 1, True)]

    cases = holes + goals
    A = [(0., 0.), (0., 1.), (1., 0.)]  # ,(-1.,0.),(0,-1.)]
    A_str = ["X", "v", "->"]  # ,"<-","^"]
    start = np.array([.5, .5])
    m = EnvGridWorld(dim, std, cases, trajectory_max_size, True, noise_type=noise_type, init_s=start, actions=A,
                           actions_str=A_str)
    emax = dim_x + dim_y + 2  # voir plus avec bruit gaussian et aller retour
    return m, emax


def generate_test_6(std=(0.5, 0.5), trajectory_max_size=None, noise_type="gaussian_bis"):
    dim = (4, 4)
    dim_x, dim_y = dim
    if trajectory_max_size is None:
        trajectory_max_size = int((dim_x + dim_y) + (dim_x + dim_y) / 2.)
    goals = [((3, 3, 4, 4), 100, 0, True)]
    holes = [((1, 0, 3, 1), 0, 1, True),
             ((1, 1, 2, 2), 0, 1, True),
             ((0, 3, 1, 4), 0, 1, True),
             ((2, 3, 3, 4), 0, 1, True)]

    cases = holes + goals
    A = [(0., 0.), (0., 1.), (1., 0.)]  # ,(-1.,0.),(0,-1.)]
    A_str = ["X", "v", "->"]  # ,"<-","^"]
    start = np.array([.5, .5])
    m = EnvGridWorld(dim, std, cases, trajectory_max_size, True, noise_type=noise_type, init_s=start, actions=A,
                           actions_str=A_str, penalty_on_move=0.)
    emax = dim_x + dim_y + 2  # voir plus avec bruit gaussian et aller retour
    return m, emax
