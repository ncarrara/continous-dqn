import numpy as np
import math
import random


def construct_random_pi():
    def pi(s,actions):
        idx = np.random.choice(len(actions), 1)[0]
        return actions[idx]
    return pi

def change_trajectories(M, trajectories):
    res = []
    for trajectory in trajectories:
        t = []
        for sample in trajectory:
            s, a, rp, sp, cp = sample
            if cp > 0.:
                rp = -M
            t.append((s, a, rp, sp, cp))
        res.append(t)
    return res


def change_samples(M, samples):
    res = []
    for sample in samples:
        s, a, rp, sp, cp = sample
        if cp > 0.:
            rp = -M
        res.append((s, a, rp, sp, cp))
    return res


def trajectories_to_samples(trajectories):
    # print("trajectories_to_samples")
    samples = []
    for trajectory in trajectories:
        samples.extend(trajectory)
    return samples


def create_arrangements(nb_elements, size_arr, current_size_arr=0, arrs=None):
    new_arrs = []
    if not arrs:
        arrs = [[]]
    for arr in arrs:
        for i in range(0, nb_elements):
            new_arr = list(arr)
            new_arr.append(i)
            new_arrs.append(new_arr)
    if current_size_arr >= size_arr - 1:
        return new_arrs
    else:
        return create_arrangements(nb_elements=nb_elements,
                                   size_arr=size_arr,
                                   current_size_arr=current_size_arr + 1,
                                   arrs=new_arrs)

def epsilon_decay(start=1.0, decay=0.01, N=100):
    return [np.exp(-n / (1. / decay)) * start for n in range(N)]

def arrToIndex(arr, size_arr, nb_elements):
    i = 0
    k = 0
    for z in arr:
        i += z * np.power(nb_elements, size_arr - k - 1)
        k += 1
    return i


def argmax(actions, state, q):
    result = float('-inf')  # -math.inf
    best_actions = []
    # print actions
    for action in actions:
        temp = q(state, action)
        if np.isnan(temp):
            print(state, action)
            raise Exception("temp = {}".format(temp))
        if temp > result:
            best_actions = [action]
            result = temp
        elif temp == result:
            best_actions.append(action)
    if not best_actions:
        print("WARN : best_actions is empty")
        best_actions = actions
    return best_actions[int(math.floor(len(best_actions) * random.random()))]


def normalized(a):
    sum = 0.
    for v in a:
        sum += v
    rez = np.array([v / sum for v in a])
    return rez
