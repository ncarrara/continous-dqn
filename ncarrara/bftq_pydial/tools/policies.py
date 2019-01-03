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
    def __init__(self, e,safeness=0.5):
        self.safeness = safeness
        self.e = e

    def ask(self, recos_status):
        not_valid_idx = []
        for idx_reco, reco in enumerate(recos_status):
            if reco is None or reco < 0.75:
                not_valid_idx.append(idx_reco)
        if len(not_valid_idx) == 0:
            return "SUMMARIZE_AND_INFORM"

        else:
            idx_ask = np.random.choice(not_valid_idx)
            if np.random.rand() < self.safeness:
                return "ASK_ORAL({})".format(idx_ask)
            else:
                return "ASK_NUM_PAD({})".format(idx_ask)

    def execute(self, s, action_mask=None, info=None):
        turn = s["turn"]
        act = None
        if turn == 0:
            act = self.ask(s["recos_status"])
        else:
            usr_action = s['str_usr_actions'][s["turn"] - 1]
            if usr_action == "DENY_SUMMARIZE":
                min_reco = np.inf
                min_idx_reco = 0
                for ireco, reco in enumerate(s["recos_status"]):
                    if reco is None:
                        min_idx_reco = ireco
                        break
                    else:
                        if reco < min_reco:
                            min_idx_reco = ireco
                            min_reco = reco
                if np.random.rand() < self.safeness:
                    act = "ASK_ORAL({})".format(min_idx_reco)
                else:
                    act = "ASK_NUM_PAD({})".format(min_idx_reco)
            else:
                act = self.ask(s["recos_status"])
        return self.e.system_actions.index(act), False, info


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
