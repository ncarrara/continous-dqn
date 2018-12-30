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


class RandomPolicy(Policy):
    def __init__(self):
        pass

    def reset(self):
        pass

    def execute(self, s, action_mask, info):
        actions = []
        for i in range(len(action_mask)):
            if action_mask[i]==0:
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
    def __init__(self, pi,  env, feature):
        self.env = env
        self.pi = pi
        self.feature = feature

    def reset(self):
        pass

    def execute(self, s, action_mask, info):
        a = self.pi(self.feature(s,self.env), action_mask)
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
    def __init__(self, pi,  env, feature):
        self.env = env
        self.feature = feature
        self.pi = pi

    def reset(self):
        pass

    def execute(self, s, action_mask, info):
        a, beta = self.pi(self.feature(s,self.env), info["beta"], action_mask)
        info["beta"] = beta
        return a, False, info
