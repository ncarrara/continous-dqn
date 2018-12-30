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
    def execute(self, s, actions, info):
        pass


# class HandcraftedPolicy(Policy):
#
#     def __init__(self, domainString="CamRestaurants"):
#         self.domainString = domainString
#         self.pi = HDCPolicy(domainString=domainString)
#         # self.env = env
#
#     def reset(self):
#         self.pi.restart()
#
#     def execute(self, s, actions, info):
#         a = self.pi.nextAction(s.getDomainState(self.domainString))
#         return a, True, info


# class EpsilonHandcraftedPolicy(Policy):
#
#     def __init__(self, domainString="CamRestaurants", epsilon=0.3):
#         self.epsilon = epsilon
#         self.pi_random = RandomPolicy()
#         self.pi_handcrafted = HandcraftedPolicy(domainString=domainString)
#
#     def reset(self):
#         self.pi_handcrafted
#
#     def execute(self, s, actions, info):
#         if np.random.random_sample() > self.epsilon:
#             a, is_master_action, info = self.pi_handcrafted.execute(s, actions, info)
#         else:
#             a, is_master_action, info = self.pi_random.execute(s, actions, info)
#         return a, is_master_action, info


class RandomPolicy(Policy):
    def __init__(self):
        pass

    def reset(self):
        pass

    def execute(self, s, actions, info):
        a = np.random.choice(range(len(actions)))
        a = actions[a]
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
    def __init__(self, pi, all_actions, env, feature):
        self.env = env
        self.pi = pi
        self.all_actions = all_actions
        self.feature = feature

    def reset(self):
        pass

    def execute(self, s, actions, info):
        action_mask = np.ones(len(self.all_actions))
        for available_action in actions:
            action_mask[self.all_actions.index(available_action)] = 0.
        a = self.pi(self.feature(s,self.env), action_mask)
        a = self.all_actions[a]
        return a, False, info


class EpsilonGreedyPolicy(Policy):
    def __init__(self, pi_greedy, epsilon):
        self.pi_greedy = pi_greedy
        self.pi_random = RandomPolicy()
        self.epsilon = epsilon

    def reset(self):
        pass

    def execute(self, s, actions, info):
        if np.random.random_sample() > self.epsilon:
            a, is_master_action, info = self.pi_greedy.execute(s, actions, info)
        else:
            a, is_master_action, info = self.pi_random.execute(s, actions, info)

        return a, is_master_action, info


class PytorchBudgetedFittedPolicy(Policy):
    def __init__(self, pi, all_actions, env, feature):
        self.env = env
        self.feature = feature
        self.pi = pi
        self.all_actions = all_actions

    def reset(self):
        pass

    def execute(self, s, actions, info):
        action_mask = np.ones(len(self.all_actions))
        for available_action in actions:
            action_mask[self.all_actions.index(available_action)] = 0.
        a, beta = self.pi(self.feature(s,self.env), info["beta"], action_mask)
        a = self.all_actions[a]
        info["beta"] = beta
        return a, False, info
