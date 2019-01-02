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


def feature_slot_filling(s,e):
    raise Exception("TODO")

