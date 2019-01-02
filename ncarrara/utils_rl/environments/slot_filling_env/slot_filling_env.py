import random

import numpy
import numpy as np
from gym.spaces import Discrete

from collections import namedtuple
import logging

from ncarrara.utils_rl.environments.slot_filling_env.user.handcrafted_user import HandcraftedUser

logger = logging.getLogger(__name__)
from enum import Enum


class Action():
    def __init__(self, label, id):
        self.label = label
        self.id = id

    def __str__(self):
        return "{}".format(self.label)

    def set_id(self, id):
        self.id = id
        return self

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(self, other.__class__):
            return self.label == other.label and self.id == other.id
        return False


# EXPL_CONF_CURRENT = Action("EXPL_CONF_CURR", 0)
ASK_NEXT = Action("ASK_NEXT", None)
REPEAT_NUMERIC_PAD = Action("REPEAT_NUMERIC_PAD", None)
REPEAT_ORAL = Action("REPEAT_ORAL", None)
SUMMARIZE_AND_INFORM = Action("SUMMARIZE_INFORM", None)
M_NONE = Action("M_NONE", None)
# WELCOME = Action("WELCOME", 4)


INFORM_CURRENT = Action("INFORM_CURR", None)
HANGUP = Action("HANGUP", None)
WTF = Action("WTF", None)
U_NONE = Action("U_NONE", None)
THANKS_BYE = Action("THANKS_BYE", None)
DENY_SUMMMARIZE = Action("DENY_SUMMMARIZE", None)

# HELLO = Action("HELLO", 5)

State = namedtuple('State',
                   ["t", "overflow_slots", "overflow_max_traj", 'reco', 'reco_by_slot', "i_slot", "current_slot",
                    "user_acts",
                    "machine_acts",
                    'others'], verbose=False)
UserObservation = namedtuple('UserObservation',
                             ["overflow_slots", "overflow_max_traj", "i_slot", "current_slot", "errs", "user_acts",
                              "machine_acts", "t"],
                             verbose=False)
CONS_ERROR = -1
CONS_VALID = 1
CONS_MISSING = 0


class SlotFillingEnv(object):

    def seed(self, seed):
        random.seed(seed)
        numpy.random.seed(seed)

    ID="slot_filling_env_v0"

    def __init__(self, user_params={"cerr": -1, "cok": 1, "ser": 0.3, "std": 0.2, "proba_handup": 0.3},
                 max_size=None,
                 size_constraints=3,
                 stop_if_summarize=True,
                 reward_if_fail=0.,
                 reward_by_turn=0.,
                 reward_if_summarize_and_fail=0.,
                 reward_if_max_trajectory=0.,
                 reward_if_uaction_is_wtf=0.,
                 reward_if_sucess=100.):
        self.reward_if_sucess = reward_if_sucess
        self.size_constraints = size_constraints
        self.reward_if_max_trajectory = reward_if_max_trajectory
        self.__user = HandcraftedUser(user_params)
        self.reward_if_fail = reward_if_fail
        self.reward_by_turn = reward_by_turn
        self.max_size = max_size
        self.reward_if_summarize_and_fail = reward_if_summarize_and_fail
        self.stop_if_summarize = stop_if_summarize
        self.reward_if_uaction_is_wtf = reward_if_uaction_is_wtf
        self.action_space = Discrete(len(self.system_actions()))

    def _get_user_(self):
        return self.__user

    def system_actions(self):
        actions = np.array([ASK_NEXT, REPEAT_NUMERIC_PAD, REPEAT_ORAL, SUMMARIZE_AND_INFORM])
        # actions = [ASK_NEXT,REPEAT_SLOWLY,REPEAT,SUMMARIZE_AND_INFORM] # TODO REDO
        for idx, action in enumerate(actions):
            action.set_id(idx)
        return actions

    def system_actions_str(self):
        return np.array([str(action) for action in self.system_actions()])

    def action_space_str(self):
        return self.system_actions_str()

    def user_actions(self):
        actions = np.array([INFORM_CURRENT, HANGUP, WTF, U_NONE, THANKS_BYE, DENY_SUMMMARIZE])  # , HELLO])
        for idx, action in enumerate(actions):
            action.set_id(idx)
        return actions

    def user_actions_str(self):
        return np.array([str(action) for action in self.user_actions()])

    def close(self):
        self.__user.close()

    def score(self):
        return

    def __str__(self):
        return str(self.__user)

    def reset(self):
        # print "----- NEW TRAJECTORY -------"
        self.__user.reset()
        self.t = 0
        self.recos = np.array([1.])
        self.reco_by_slot = np.full((self.size_constraints,), 0.)
        self.errs = np.array([None])
        self.user_acts = np.array([U_NONE])
        self.machine_acts = np.array([M_NONE])
        self.slots = np.full((self.size_constraints,), CONS_MISSING)
        self.current_slot = -1
        self.i_slot = -1
        self.overflow_slots = False
        self.overflow_max_traj = False
        self.state = State(self.t, self.overflow_slots, self.overflow_max_traj, self.recos[-1],
                           np.copy(self.reco_by_slot),
                           self.i_slot, self.current_slot,
                           self.user_acts, self.machine_acts, {
                               "state_is_absorbing": False,
                               "errs": self.errs})
        observation = self.state
        self.nb_reset_current_slot = 0
        self.last_inform_reco = None

        return observation

    def increment_current_slot(self):
        self.i_slot += 1
        self.current_slot = self.i_slot % len(self.slots)
        self.overflow_slots = self.current_slot == len(self.slots) - 1

    def reset_current_slot(self):
        self.i_slot = -1
        self.overflow_slots = False  # "ON va overflow si on a asknext" et non "on a overflow"
        self.current_slot = -1
        self.nb_reset_current_slot += 1.
        self.reco_by_slot = np.full((self.size_constraints,), 0.)
        self.last_inform_reco = None
        self.slots = np.full((self.size_constraints,), CONS_MISSING)

    def step(self, action):
        # print "slots : ", self.slots
        # print "current_slot : ", self.current_slot
        # print "system says :", action
        end_from_user = False
        reco = 1.
        err = None
        s_action = action
        self.machine_acts = np.append(self.machine_acts, action)

        success = False
        fail = False
        u_action = None
        other_info_user = {"user_is_pissed_of_by_repeat": False, "user_is_pissed_of_by_implconf": False}
        if s_action == SUMMARIZE_AND_INFORM:
            if np.all(self.slots == CONS_VALID):
                success = True
                u_action = THANKS_BYE
            elif np.any(self.slots == CONS_MISSING):
                success = False
                u_action = WTF
                self.slots = np.full((self.size_constraints,), CONS_MISSING)
                self.reco_by_slot = np.full((self.size_constraints,), 0.)
            else:
                success = False
                u_action = DENY_SUMMMARIZE
                self.slots = np.full((self.size_constraints,), CONS_MISSING)
                self.reco_by_slot = np.full((self.size_constraints,), 0.)
            self.reset_current_slot()
        elif self.current_slot == -1 and (s_action == REPEAT_NUMERIC_PAD or s_action == REPEAT_ORAL):
            self.slots = np.full((self.size_constraints,), CONS_MISSING)
            self.reco_by_slot = np.full((self.size_constraints,), 0.)
            self.reset_current_slot()
            u_action = WTF
        elif s_action == ASK_NEXT and self.overflow_slots:
            u_action = WTF
        else:
            if s_action == ASK_NEXT:
                self.increment_current_slot()

            obs = UserObservation(self.overflow_slots, self.overflow_max_traj, self.i_slot, self.current_slot,
                                  self.errs, self.user_acts,
                                  self.machine_acts, self.t)
            u_action, other_info_user = self.__user.action(obs)

        if  u_action == INFORM_CURRENT:

            if s_action == REPEAT_NUMERIC_PAD:
                err = False
                reco = 1.
            elif s_action == REPEAT_ORAL:
                err = np.random.rand() < self.__user.ser
                reco = self.__gen_error_reco(err=err, cerr=self.__user.cerr, cok=self.__user.cok,
                                             std=self.__user.cstd)

                # err = np.random.rand() < self.__user.ser #False
                # reco = 1. if not err else 0.

                # err = np.random.rand() < self.__user.ser/2.  # False
                # reco = self.__gen_error_reco(err=err, cerr=self.__user.cerr, cok=self.__user.cok)

            elif s_action == ASK_NEXT:
                err = np.random.rand() < self.__user.ser
                reco = self.__gen_error_reco(err=err, cerr=self.__user.cerr, cok=self.__user.cok,
                                             std=self.__user.cstd)
            else:
                raise Exception("whhhat? {}".format(s_action))
            self.reco_by_slot[self.current_slot] = reco
            self.slots[self.current_slot] = CONS_ERROR if err else CONS_VALID

            # print reco

        self.recos = np.append(self.recos, reco)
        self.errs = np.append(self.errs, err)
        fail = u_action == HANGUP
        self.user_acts = np.append(self.user_acts, u_action)
        # print "user says : ", u_action

        self.t += 1
        end_from_max_trajectory = (self.max_size is not None and self.t >= self.max_size)
        self.overflow_max_traj = (self.max_size is not None and self.t >= self.max_size - 1)

        if success:
            reward = self.reward_if_sucess
        elif fail:
            reward = self.reward_if_fail
        elif end_from_max_trajectory:
            reward = self.reward_if_max_trajectory
        elif u_action == WTF:
            reward = self.reward_if_uaction_is_wtf
        else:
            if self.machine_acts[-1] == SUMMARIZE_AND_INFORM:
                reward = self.reward_if_summarize_and_fail
            else:
                reward = self.reward_by_turn
            self.__user.update(self)

        if self.stop_if_summarize:
            end = u_action == WTF or end_from_max_trajectory or success or fail or (
                    self.machine_acts[-1] == SUMMARIZE_AND_INFORM and not success)
        else:
            end = end_from_max_trajectory or success or fail  # TODO REDO

        if self.t > 1000:
            logger.warning("[WARNING] dialogue size is {} with user {}".format(self.t, self.__user))

        state_ = State(self.t, self.overflow_slots, self.overflow_max_traj, reco, np.copy(self.reco_by_slot),
                       self.i_slot, self.current_slot,
                       self.user_acts,
                       self.machine_acts,
                       {"state_is_absorbing": end,
                        "errs": self.errs})
        # if fail:
        #     print state_

        observation, reward, done, info = state_, reward, end, {"success": success,
                                                                "state_is_absorbing": end,
                                                                "fail": fail,
                                                                "user_is_pissed_of_by_repeat": other_info_user[
                                                                    "user_is_pissed_of_by_repeat"],
                                                                "user_is_pissed_of_by_implconf": other_info_user[
                                                                    "user_is_pissed_of_by_implconf"]
                                                                }
        return observation, reward, done, info

    def __gen_error_reco(self, err, cerr=-1, cok=1, std=0.2):

        reco_err = 1. / (1 + np.exp(-np.random.normal(cerr, std)))
        reco_sucess = 1. / (1 + np.exp(-np.random.normal(cok, std)))
        return reco_err if err else reco_sucess
