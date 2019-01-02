def feature_factory(feature_str):
    if feature_str == "feature_pydial":
        return feature_pydial
    elif feature_str == "feature_slot_filling":
        return feature_slot_filling


def feature_pydial(s, e):
    N_actions = e.action_space.n
    if s is None:
        return None
    else:
        # pprint.pprint(s)
        summary_acts = s["system_summary_acts"]
        master_acts = s["system_master_acts"]
        user_acts = s["user_acts"]

        flatten = s["flatten"]
        last_act_onehot = [0.] * (N_actions)  # + 1)
        repeat_one_hot = [0.] * 11

        # what is my last summary act
        if not master_acts:
            pass
        else:
            last_act_onehot[e.action_space_str.index(summary_acts[-1])] = 1.

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


# def feature_slot_filling(s, e):
#     if s is None:
#         return None
#     reco = s["reco_by_slot"]
#     one_hot_current_slot = [0.] * e.size_constraints
#     one_hot_current_slot[s["current_slot"]] = 1.
#
#     one_hot_sys_act = [0.] * len(e.system_actions)
#     if len(s["machine_acts"]) > 0:
#         one_hot_sys_act[e.system_actions.index(s["machine_acts"][-1])] = 1.
#
#     one_hot_usr_act = [0.] * len(e.user_actions)
#     if len(s["user_acts"]) > 0:
#         one_hot_usr_act[e.user_actions.index(s["user_acts"][-1])] = 1.
#
#     feat = reco + one_hot_current_slot + one_hot_sys_act + one_hot_usr_act
#     # print(feat)
#     print("reco={} current_slot={} sys_act={} usr_act={}".format("".join(["{:.2f}".format(x) for x in reco]),
#           one_hot_current_slot, one_hot_sys_act, one_hot_usr_act))
#     return feat

# def feature_slot_filling(s, e):
#     if s is None:
#         return None
#     else:
#         one_hot_current_slot = [0.] * e.size_constraints
#         current_slot = s["current_slot"]
#         if s["current_slot"] is not None and s["current_slot"] >= 0:
#             one_hot_current_slot[current_slot] = 1.
#
#         reco = 0.0 if s["reco"] is None else s["reco"]
#         overflow = s["overflow_slots"]
#
#         one_hot_usr_act = [0.] * len(e.user_actions)
#         if len(s["user_acts"]) > 0:
#             one_hot_usr_act[e.user_actions.index(s["user_acts"][-1])] = 1.
#
#         feat = [reco] + [overflow] + one_hot_usr_act + one_hot_current_slot
#         # print("reco={:.2f} overflow={} one_hot_usr_act={} one_hot_current_slot={} ".format(reco, overflow,
#         #                                                                                    one_hot_usr_act,
#         #                                                                                    one_hot_current_slot))
#         return feat
import numpy as np


def feature_slot_filling(s, e):
    offset = e.size_constraints * (e.size_constraints + 1) / 2
    feat = np.zeros(offset + len(e.user_actions) - 1)
    current_slot = s["current_slot"]
    if current_slot >= 0 and s["user_acts"][-1] == 0:  # INFORM_CURRENT
        i = 0 if current_slot == 0 else current_slot * (current_slot + 1) / 2
        i_ = i + current_slot + 1
        recos = s["reco_by_slot"][0:current_slot + 1]
        feat[i:i_] = recos
    else:
        feat[offset + s["user_acts"][-1].id - 1] = 1.
    rez = np.concatenate((np.array([1.]), feat))
    xxx = ""
    for ss in rez:
        xxx += " {:.2f}".format(ss)
    return rez
