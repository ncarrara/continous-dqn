from env.slot_filling_2.env_slot_filling import *


def random_policy(action_space):
    def policy(s, _):
        a = np.random.choice(action_space)
        return a

    return policy


def handcrafted_policy(eps=0.0, safeness=0.5):
    def policy(s, action_space):
        user_action = s.user_acts[-1]
        if np.random.rand() < eps:
            action = np.random.choice(action_space)
        elif s.overflow_slots:
            if s.reco > 0.5:
                action = SUMMARIZE_AND_INFORM
            else:
                if np.random.rand() < safeness:
                    action = REPEAT_ORAL
                else:
                    action = REPEAT_NUMERIC_PAD
        else:
            if user_action == INFORM_NEXT or user_action == INFORM_CURRENT:
                if s.reco > 0.5:
                    action = ASK_NEXT
                elif s.reco >= 0.0:
                    if np.random.rand() < safeness:
                        action = REPEAT_ORAL
                    else:
                        action = REPEAT_NUMERIC_PAD
                else:
                    raise Exception("reco < 0")
            elif user_action == DENY_SUMMMARIZE:
                action = ASK_NEXT
            elif user_action == U_NONE:
                action = ASK_NEXT
            elif user_action == WTF:
                action = ASK_NEXT
            else:
                action = ASK_NEXT
        return action

    return policy
