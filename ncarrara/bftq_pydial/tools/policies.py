import abc
import numpy as np


# from policy.HDCPolicy import HDCPolicy


class Policy:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reset(self):
        pass

    # must return Q function (s,a) -> double
    @abc.abstractmethod
    def execute(self, s, action_mask, info):
        pass


class HandcraftedSlotFillingEnv(Policy):
    def __init__(self, safeness, system_actions):
        self.safeness = safeness
        self.system_actions = system_actions

    def execute(self, s, action_mask=None, info=None):
        user_action = None if len(s["user_acts"]) == 0 else s["user_acts"][-1]
        if user_action is None:
            action = "ASK_NEXT"
        else:
            if s["overflow_slots"]:
                if s["reco"] > 0.5:
                    action = "SUMMARIZE_AND_INFORM"
                else:
                    if np.random.rand() < self.safeness:
                        action = "REPEAT_ORAL"
                    else:
                        action = "REPEAT_NUMERIC_PAD"
            else:
                if user_action == "INFORM_CURRENT":
                    if s["reco"] > 0.5:
                        action = "ASK_NEXT"
                    elif s["reco"] >= 0.0:
                        if np.random.rand() < self.safeness:
                            action = "REPEAT_ORAL"
                        else:
                            action = "REPEAT_NUMERIC_PAD"
                    else:
                        raise Exception("reco < 0")
                elif user_action == "DENY_SUMMMARIZE":
                    action = "ASK_NEXT"
                elif user_action == "U_NONE":
                    action = "ASK_NEXT"
                elif user_action == "WTF":
                    action = "ASK_NEXT"
                else:
                    raise Exception("This is not possible (user_action={})".format(user_action))
        i_action = self.system_actions.index(action)
        return i_action,None,{}


class RandomPolicy(Policy):
    def __init__(self):
        pass

    def reset(self):
        pass

    def execute(self, s, action_mask, info):
        actions = []
        for i in range(len(action_mask)):
            if action_mask[i] == 0:
                actions.append(i)
        a = np.random.choice(actions)
        return a, False, info


class FittedPolicy(Policy):
    def __init__(self, pi):
        self.pi = pi

    def reset(self):
        pass

    def execute(self, s, actions, info):
        a = self.pi(s, actions)
        return a, False, info


class PytorchFittedPolicy(Policy):
    def __init__(self, pi, env, feature):
        self.env = env
        self.pi = pi
        self.feature = feature

    def reset(self):
        pass

    def execute(self, s, action_mask, info):
        a = self.pi(self.feature(s, self.env), action_mask)
        return a, False, info


class EpsilonGreedyPolicy(Policy):
    def __init__(self, pi_greedy, epsilon):
        self.pi_greedy = pi_greedy
        self.pi_random = RandomPolicy()
        self.epsilon = epsilon

    def reset(self):
        pass

    def execute(self, s, action_mask, info):
        if np.random.random_sample() > self.epsilon:
            a, is_master_action, info = self.pi_greedy.execute(s, action_mask, info)
        else:
            a, is_master_action, info = self.pi_random.execute(s, action_mask, info)

        return a, is_master_action, info


class PytorchBudgetedFittedPolicy(Policy):
    def __init__(self, pi, env, feature):
        self.env = env
        self.feature = feature
        self.pi = pi

    def reset(self):
        pass

    def execute(self, s, action_mask, info):
        a, beta = self.pi(self.feature(s, self.env), info["beta"], action_mask)
        info["beta"] = beta
        return a, False, info
