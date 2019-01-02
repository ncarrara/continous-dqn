import env.env as env
from env.slot_filling_2.env_slot_filling import *




def feature_classic(s, size_user_actions, size_system_actions, norm=False):
    machine_acts = np.zeros(size_system_actions)
    user_acts = np.zeros(size_user_actions)
    # print s.machine_acts
    for act in s.machine_acts:
        if act is not None:
            machine_acts[act.id] += 1.
    # print s.user_acts
    for act in s.user_acts:
        if act is not None:
            # print act
            user_acts[act.id] += 1.
    others_features = np.array([float(s.reco), float(s.t)])
    # machine_acts = machine_acts if sum(machine_acts)==0. else machine_acts/np.linalg.norm(machine_acts)
    # user_acts = user_acts if sum(user_acts) ==0. else machine_acts/np.linalg.norm(user_acts)
    arr = np.concatenate((machine_acts, user_acts, others_features))
    if norm:
        return arr / np.linalg.norm(arr)  # TODO pas de sens de norme vu que l'on compte
    else:
        return arr


def feature_classic_6(s, size_user_actions, size_system_actions):
    machine_acts = np.zeros(size_system_actions)
    user_acts = np.zeros(size_user_actions)
    # print s.machine_acts
    for act in s.machine_acts:
        if act is not None:
            machine_acts[act.id] += 1.
    for act in s.user_acts:
        if act is not None:
            # print act
            user_acts[act.id] += 1.
    others_features = np.array([float(s.reco), float(s.t)])
    arr = np.concatenate((s.others["slots_C"], s.others["slots_R"], others_features))
    return arr
    # return arr / np.linalg.norm(arr)


def feature_classic_2(s, size_user_actions, size_system_actions):
    other = np.array([float(s.reco), float(s.t)])
    return other


def feature_id(s):
    return s


def create_gaussian_centers(self):
    n = self.size_constraints * (self.size_constraints + 1) / 2
    n_feature = self.size_user_actions * self.size_constraints + n
    print("size feature", n_feature)
    n_centers = np.power(2, self.size_constraints + 1) - 2 + self.size_user_actions * self.size_constraints + 1
    centers = np.zeros(n_feature)
    n_centers = 0
    for slot in range(0, self.size_constraints):
        offset = self.size_user_actions * slot + (slot + 1) * (slot) / 2
        # print i_slot
        codings = self.create_coding(slot + 1)
        for coding in codings:
            center = np.zeros(n_feature)
            center[offset:offset + slot + 1] = coding
            center[offset + slot + 1 + 0] = 1.
            # centers[k] = center
            centers = np.vstack((centers, center))
            n_centers += 1
        for i in range(1, self.size_user_actions):
            center = np.zeros(n_feature)
            center[offset + slot + 1 + i] = 1.
            # centers[k] = center
            n_centers += 1
            centers = np.vstack((centers, center))
    return centers


def feature_simple_not_markov(self, s):
    # print s.all_recos
    # c'est pas markov !!!!
    u = np.zeros(self.size_user_actions)
    # sys = np.zeros(self.size_system_actions+1)
    u[s.user_acts[-1].id] = 1.
    # print s.machine_acts[-1].label
    # sys[s.machine_acts[-1].id] = 1.
    feat = u * s.reco
    rez = np.zeros(len(feat) * 2)
    if s.overflow_slots:
        rez[:len(feat)] = feat
    else:
        rez[len(feat):] = feat
    rez = np.concatenate((np.array([1.]), rez))
    # print rez
    # eventuellement la partie s.over_flow_max_traj
    return rez


def feature_simple_markov(self, s):
    offset = self.size_constraints * (self.size_constraints + 1) / 2
    feat = np.zeros(offset + self.size_user_actions - 1)
    current_slot = s.current_slot
    if current_slot >= 0 and s.user_acts[-1] == INFORM_CURRENT:
        i = 0 if current_slot == 0 else current_slot * (current_slot + 1) / 2
        i_ = i + current_slot + 1
        recos = s.reco_by_slot[0:current_slot + 1]
        feat[i:i_] = recos
    else:
        feat[offset + s.user_acts[-1].id - 1] = 1.
    rez = np.concatenate((np.array([1.]), feat))
    xxx = ""
    for ss in rez:
        xxx += " {:.2f}".format(ss)
    return rez


def lambda_rbf_quick(s, centers, sigma):
    exp = -1. / 2. * np.power(np.linalg.norm(np.array(s) - np.array(centers), axis=1) / sigma, 2)
    gauss = (1. / (sigma * np.sqrt(2. * np.pi))) * np.exp(exp)
    return gauss


def feature_rbf(self, s):
    feat = np.zeros(self.size_constraints + 3)
    if s.user_acts[-1] == INFORM_CURRENT:
        feat[0:self.size_constraints] = s.reco_by_slot
    else:
        feat[self.size_constraints + (s.user_acts[-1].id - 1)] = 1.

    feat = self.lambda_rbf_quick(feat, self.centers, self.sigma_rbf)
    print("---------------")

    return np.concatenate((np.array([1.]), feat))


def feature_simple(self, s):
    if s.current_slot < 0:
        rbfvector = np.zeros(self.n_rbf_vector)
        offset = self.size_user_actions * s.current_slot + (s.current_slot + 1) * (s.current_slot) / 2
        if s.user_acts[-1] == INFORM_CURRENT:
            rbfvector[offset:offset + s.current_slot] = s.reco_by_slot[0:s.current_slot]
        rbfvector[offset + s.current_slot + s.user_acts[-1].id] = 1.
        feat = self.lambda_rbf_quick(rbfvector, self.centers, self.sigma_rbf)
        # print ["{:.2f}".format(v) for v in feat]
    else:
        feat = np.zeros(self.n_features)

    return feat


def create_coding(n):
    codings = np.zeros((np.power(2, n), n))
    for i in range(1 << n):
        s = bin(i)[2:]
        s = '0' * (n - len(s)) + s
        codings[i] = map(int, list(s))
    print(codings)
    return codings


def create_gaussian_centers_2(size_constraints):
    codings = create_coding(size_constraints)
    centers = np.array(codings)
    return centers


def feature_rbf_2(s, centers, size_constraints, size_user_actions, sigma_rbf):
    feat = np.zeros(size_constraints * len(centers) + size_user_actions)
    if s.user_acts[-1] == INFORM_CURRENT:
        reco = lambda_rbf_quick(s.reco_by_slot, centers, sigma_rbf)
        offset = s.current_slot * len(centers)
        feat[offset:offset + len(centers)] = reco
    feat[size_constraints * len(centers) + s.user_acts[-1].id] = 1.
    feat = np.concatenate((np.array([1.]), feat))
    return feat

def feature_rbf_2_no_constant(s, centers, size_constraints, size_user_actions, sigma_rbf):
    feat = np.zeros(size_constraints * len(centers) + size_user_actions)
    if s.user_acts[-1] == INFORM_CURRENT:
        reco = lambda_rbf_quick(s.reco_by_slot, centers, sigma_rbf)
        offset = s.current_slot * len(centers)
        feat[offset:offset + len(centers)] = reco
    feat[size_constraints * len(centers) + s.user_acts[-1].id] = 1.
    # feat = np.concatenate((np.array([1.]), feat))
    return feat


def feature_tree_2(s):
    feat = np.array([s.user_acts[-1].id, s.reco_by_slot[-1], s.current_slot])
    return feat
