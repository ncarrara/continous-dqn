import numpy as np
import pprint


def build_feature_autoencoder(info):
    feature_str = info["feature_str"]
    if feature_str == "feature_autoencoder_identity":
        return feature_autoencoder_identity
    elif feature_str == "feature_autoencoder_pydial":
        return lambda transition: feature_autoencoder_pydial(transition, info["action_str"])
    elif feature_str == "feature_autoencoder_slot_filling":
        return lambda transition: feature_autoencoder_slot_filling(transition, info["e"])

    else:
        raise Exception("Unknown feature : {}".format(feature_str))


def build_feature_dqn(info):
    feature_str = info["feature_str"]
    if feature_str == "feature_dqn_identity":
        return feature_dqn_identity
    elif feature_str == "feature_dqn_pydial":
        return lambda transition: feature_dqn_pydial(transition, info["action_str"])
    else:
        raise Exception("Unknown feature : {}".format(feature_str))


def feature_autoencoder_identity(transition):
    s, a, r, s_, done, info = transition
    if type(s) == type(np.zeros(0)):
        s = s.tolist()
        s_ = s_.tolist()
    if s_ is None:
        s_ = [0.0] * len(s)
    rez = s + [a] + [r] + s_
    return rez

from ncarrara.budgeted_rl.tools.features import feature_basic
def feature_autoencoder_slot_filling(transition, e):
    s, a, r, s_, done, info = transition
    s=feature_basic(s)
    s_=feature_basic(s_)
    return s + [a] + [r] + s_


def feature_autoencoder_pydial(transition, action_space_str):
    s, a, r, s_, done, info = transition

    s = feature_dqn_pydial(s, action_space_str)
    if s_ is None:
        s_ = [0.0] * len(s)
    else:
        s_ = feature_dqn_pydial(s_, action_space_str)

    return s + [a] + [r] + s_


def feature_dqn_identity(s):
    # je suis sur de vouloir faire ça ????
    if type(s) == type(np.zeros(1)):
        s = s.tolist()
    return s


def feature_dqn_pydial(s, action_space_str):
    if s is None:
        return None
    else:
        summary_acts = s["system_summary_acts"]
        master_acts = s["system_master_acts"]
        user_acts = s["user_acts"]

        flatten = s["flatten"]
        last_act_onehot = [0.] * (len(action_space_str))
        repeat_one_hot = [0.] * 11

        # what is my last summary act
        if not master_acts:
            pass
        else:
            last_act_onehot[action_space_str.index(summary_acts[-1])] = 1.

        # how many I ask twice the same thing
        concecutive_repetion = 0
        if not master_acts:
            pass
        else:
            i = 0
            while i < len(master_acts) - 1:
                if (master_acts[i] == master_acts[i + 1] and (
                        "repeat" not in user_acts[i])):  # or "null" in acts[i] or "badact" in acts[i]:
                    concecutive_repetion += 1
                i += 1
            repeat_one_hot[concecutive_repetion] = 1.

        rez = flatten + last_act_onehot + repeat_one_hot

        return rez
