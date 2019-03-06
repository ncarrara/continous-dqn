import numpy as np


def build_feature_autoencoder(info):
    feature_str = info["feature_str"]
    if feature_str == "feature_autoencoder_identity":
        return feature_autoencoder_identity
    elif feature_str == "feature_autoencoder_slot_filling":
        size_constraints = info["size_constraints"]
        max_turn = info["max_turn"]
        user_actions = info["user_actions"]
        system_actions = info["system_actions"]
        return lambda transition, device: feature_autoencoder_slot_filling(transition, device, size_constraints,
                                                                           max_turn,
                                                                           user_actions,
                                                                           system_actions)

    else:
        raise Exception("Unknown feature : {}".format(feature_str))


def build_feature_dqn(info):
    feature_str = info["feature_str"]
    if feature_str == "feature_dqn_identity":
        return feature_dqn_identity
    elif feature_str == "feature_basic":
        size_constraints = info["size_constraints"]
        max_turn = info["max_turn"]
        user_actions = info["user_actions"]
        system_actions = info["system_actions"]
        return lambda transition: feature_basic(transition, size_constraints, max_turn, user_actions, system_actions)
    else:
        raise Exception("Unknown feature : {}".format(feature_str))


def feature_autoencoder_identity(transition, device):
    s, a, r, s_, done, info = transition
    import torch
    if type(s) == type(torch.zeros(0)):
        if s_ is None:
            s_ = torch.zeros(s.shape).to(device)
        rez = torch.cat([s, a.unsqueeze(0).float(), r.unsqueeze(0), s_], dim=len(s.shape) - 1)
    else:
        if type(s) == type(np.zeros(0)):
            s = s.tolist()
            s_ = s_.tolist()
        if s_ is None:
            s_ = [0.] * len(s)
        rez = s + [a] + [r] + s_
    return rez


def feature_dqn_identity(s):
    # je suis sur de vouloir faire ça ????
    if type(s) == type(np.zeros(1)):
        s = s.tolist()
    return s


############################ SLOT FILLING ###############################"

def feature_autoencoder_slot_filling(transition, device, size_constraints, max_turn, user_actions, system_actions):
    s, a, r, s_, done, info = transition
    import torch
    if type(s) == type(torch.zeros(0)):
        if s_ is None:
            s_ = torch.zeros(s.shape).to(device)
        else:
            s_ = feature_basic(s_.cpu().numpy().tolist(), size_constraints, max_turn, user_actions, system_actions)
            s_ = torch.tensor(s_, device=device)

        s = feature_basic(s.cpu().numpy().tolist(), size_constraints, max_turn, user_actions, system_actions)
        s = torch.tensor(s, device=device)

        rez = torch.cat([s, a.unsqueeze(0).float(), r.unsqueeze(0), s_], dim=len(s.shape) - 1)
    else:
        if type(s) == type(np.zeros(0)):
            s = s.tolist()
            s_ = s_.tolist()
        s = feature_basic(s, size_constraints, max_turn, user_actions, system_actions)
        s_ = feature_basic(s_, size_constraints, max_turn, user_actions, system_actions)
        if s_ is None:
            s_ = [0.] * len(s)
        rez = s + [a] + [r] + s_
    return rez


def feature_basic(s, size_constraints, max_turn, user_actions, system_actions):
    if s is None:
        return None
    else:
        reco = feat_reco_status(s, size_constraints)
        min_reco = [0] * len(reco)
        min_reco[np.argmin(np.asarray(reco))] = 1.
        return min_reco + reco + feat_turn(s, max_turn) + feat_usr_act(s, user_actions) + feat_sys_act(s,
                                                                                                       system_actions)


def feat_turn(s, max_turn):
    one_hot_turn = [0] * max_turn
    one_hot_turn[s["turn"]] = 1
    return one_hot_turn


def feat_sys_act(s, system_actions):
    one_hot_sys_act = [0.] * len(system_actions)
    if s["turn"] == 0:
        pass
    else:
        one_hot_sys_act[system_actions.index(s["str_sys_actions"][s["turn"] - 1])] = 1.
    return one_hot_sys_act


def feat_usr_act(s, user_actions):
    one_hot_usr_act = [0.] * len(user_actions)
    if s["turn"] == 0:
        pass
    else:
        one_hot_usr_act[user_actions.index(s["str_usr_actions"][s["turn"] - 1])] = 1.
    return one_hot_usr_act


def feat_reco_status(s, size_constraints):
    if s["turn"] == 0:
        recos_status = [0.] * size_constraints
    else:
        recos_status_numpy = np.nan_to_num(np.array(s["recos_status"], dtype=np.float))
        recos_status = recos_status_numpy.tolist()
    return recos_status
