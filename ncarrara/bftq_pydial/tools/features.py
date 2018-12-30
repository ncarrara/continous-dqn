

def feature_0(s, e):
    return feature_4(s, e, False)


def feature_4(s, e, use_dontdothat=True):
    N_actions = e.action_space.n
    if s is None:
        return None
    else:
        # pprint.pprint(s)
        summary_acts = s["system_summary_acts"]
        master_acts = s["system_master_acts"]
        user_acts = s["user_acts"]

        flatten = s["flatten"]
        last_act_onehot = [0.] * (N_actions )#+ 1)
        repeat_one_hot = [0.] * 11
        dontdothat = [0.] * e.action_space.n

        # what is my last summary act
        if not master_acts:
            pass
        # elif master_acts[-1] == u'hello()':
        #     last_act_onehot[N_actions] = 1.
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

        # what summary act will repeat the same MASTER act ?
        # WARNING: take a shitload of computational time
        if use_dontdothat:
            if not master_acts:
                pass
            else:
                belief = s["pydial_state"]
                last_master_act = master_acts[-1]
                for iact, act in enumerate(e.action_space):
                    next_master_act = e._summary_act_to_master_act(belief, act, last_master_act)
                    if next_master_act == last_master_act:
                        dontdothat[iact] = 1.
        if use_dontdothat:

            rez = flatten + last_act_onehot + repeat_one_hot + dontdothat
        else:
            rez = flatten + last_act_onehot + repeat_one_hot

        return rez
