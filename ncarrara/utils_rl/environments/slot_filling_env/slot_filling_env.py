import random
from gym.spaces import Discrete
import numpy as np
from collections import namedtuple
import logging

logger = logging.getLogger(__name__)

UserObservation = namedtuple('UserObservation',
                             ["overflow_slots", "overflow_max_traj", "i_slot", "current_slot", "errs", "user_acts",
                              "machine_acts", "t"])

State = namedtuple('State',
                   ["t", "overflow_slots", "overflow_max_traj", 'reco', 'reco_by_slot', "i_slot", "current_slot",
                    "user_acts",
                    "machine_acts",
                    'others'])

CONS_ERROR = -1
CONS_VALID = 1
CONS_MISSING = 0


class SlotFillingEnv(object):

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    ID = "slot_filling_env_v0"

    def __init__(self, user_params={"cerr": -1, "cok": 1, "ser": 0.3, "cstd": 0.2, "proba_hangup": 0.3},
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
        from ncarrara.utils_rl.environments.slot_filling_env.user.handcrafted_user import HandcraftedUser
        self.__user = HandcraftedUser(**user_params)
        self.reward_if_fail = reward_if_fail
        self.reward_by_turn = reward_by_turn
        self.max_size = max_size
        self.reward_if_summarize_and_fail = reward_if_summarize_and_fail
        self.stop_if_summarize = stop_if_summarize
        self.reward_if_uaction_is_wtf = reward_if_uaction_is_wtf

        self.system_actions = ["REPEAT_NUMERIC_PAD", "REPEAT_ORAL", "SUMMARIZE_AND_INFORM"]
        self.action_space = Discrete(len(self.system_actions))
        self.action_space_str = self.system_actions
        self.user_actions = ["INFORM_CURRENT", "HANGUP", "WTF", "THANKS_BYE", "DENY_SUMMARIZE"]

    def _get_user_(self):
        return self.__user

    def action_space_str(self):
        return [str(action) for action in self.system_actions]

    def close(self):
        self.__user.close()

    def score(self):
        return

    def __str__(self):
        return str(self.__user)

    def reset(self):
        logger.info(("--------------- new dialogue -------------"))
        self.__user.reset()
        self.t = 0
        self.recos = []
        self.reco_by_slot = self.size_constraints * [None]
        reco = None
        self.errs = []
        self.user_acts = []
        self.machine_acts = []
        self.slots = self.size_constraints * [CONS_MISSING]
        self.current_slot = 0
        self.i_slot = 0
        self.will_overflow_slots = False
        self.overflow_max_traj = False
        self.state = State(self.t,
                           self.will_overflow_slots,
                           self.overflow_max_traj,
                           reco,
                           self.reco_by_slot,
                           self.i_slot,
                           self.current_slot,
                           self.user_acts, self.machine_acts, {
                               "state_is_absorbing": False,
                               "errs": self.errs})
        observation = self.state._asdict()
        self.nb_reset_current_slot = 0
        self.last_inform_reco = None

        # logger.info("recos : [{}]".format("".join(["{:.2f} ".format(x) for x in self.recos])))
        # logger.info("errs : [{}]".format("".join(["None " if x is None else "{:.2f} ".format(x) for x in self.errs])))
        # logger.info("slots : [{}]".format("".join(["None " if x is None else "{:.2f} ".format(x) for x in self.slots])))

        return observation

    def increment_current_slot(self):
        self.i_slot += 1
        self.current_slot = self.i_slot % len(self.slots)
        self.will_overflow_slots = self.current_slot == len(self.slots)

    def reset_current_slot(self):
        self.i_slot =0
        self.will_overflow_slots = False  # "ON va overflow si on a asknext" et non "on a overflow"
        self.current_slot = 0
        self.nb_reset_current_slot += 1.
        self.reco_by_slot = self.size_constraints * [0.]
        self.last_inform_reco = None
        self.slots = self.size_constraints * [CONS_MISSING]

    def step(self, action):
        print("----")
        logger.info("current slot : {}".format(self.current_slot))
        logger.info("recos : [{}]".format("".join(["{:.2f} ".format(x) for x in self.recos])))
        logger.info("errs : [{}]".format("".join(["None " if x is None else "{:.2f} ".format(x) for x in self.errs])))
        logger.info("slots state: [{}]".format("".join(["None " if x is None else "{:.2f} ".format(x) for x in self.slots])))
        action = self.system_actions[action]
        logger.info("system says : {}".format(action))
        reco = 1.
        err = None
        s_action = action
        self.machine_acts.append(action)

        success = False
        if s_action == "SUMMARIZE_AND_INFORM":
            if np.all(np.asarray(self.slots) == CONS_VALID):
                success = True
                u_action = "THANKS_BYE"
            elif np.any(np.asarray(self.slots) == CONS_MISSING):
                success = False
                u_action = "DENY_SUMMARIZE"
            else:
                success = False
                u_action = "DENY_SUMMARIZE"
            if not self.stop_if_summarize:
                self.reset_current_slot()
        elif self.current_slot == 0 and (s_action == "REPEAT_NUMERIC_PAD" or s_action == "REPEAT_ORAL"):
            self.slots = self.size_constraints * [CONS_MISSING]
            self.reco_by_slot = self.size_constraints * [0.]
            self.reset_current_slot()
            u_action = "WTF"
        elif s_action == "ASK_NEXT" and self.will_overflow_slots:
            u_action = "WTF"
        else:
            if s_action == "ASK_NEXT":
                self.increment_current_slot()

            obs = UserObservation(self.will_overflow_slots, self.overflow_max_traj, self.i_slot, self.current_slot,
                                  self.errs, self.user_acts,
                                  self.machine_acts, self.t)
            u_action, other_info_user = self.__user.action(obs)

        logger.info("user says : {}".format(u_action))

        if u_action == "INFORM_CURRENT":

            if s_action == "REPEAT_NUMERIC_PAD":
                err = False
                reco = 1.
            elif s_action == "REPEAT_ORAL":
                err = np.random.rand() < self.__user.ser
                reco = self.__gen_error_reco(err=err, cerr=self.__user.cerr, cok=self.__user.cok,
                                             std=self.__user.cstd)
            elif s_action == "ASK_NEXT":
                err = np.random.rand() < self.__user.ser
                reco = self.__gen_error_reco(err=err, cerr=self.__user.cerr, cok=self.__user.cok,
                                             std=self.__user.cstd)
            else:
                raise Exception("whhhat? {}".format(s_action))
            self.reco_by_slot[self.current_slot] = reco
            self.slots[self.current_slot] = CONS_ERROR if err else CONS_VALID

        self.recos.append(reco)

        self.errs.append(err)


        fail = u_action == "HANGUP"
        self.user_acts.append(u_action)

        self.t += 1
        end_from_max_trajectory = (self.max_size is not None and self.t >= self.max_size)
        self.overflow_max_traj = (self.max_size is not None and self.t >= self.max_size - 1)

        if success:
            reward = self.reward_if_sucess
        elif fail:
            reward = self.reward_if_fail
        elif end_from_max_trajectory:
            reward = self.reward_if_max_trajectory
        elif u_action == "WTF":
            reward = self.reward_if_uaction_is_wtf
        else:
            if self.machine_acts[-1] == "SUMMARIZE_AND_INFORM":
                reward = self.reward_if_summarize_and_fail
            else:
                reward = self.reward_by_turn
            self.__user.update(self)

        if self.stop_if_summarize:
            done = u_action == "WTF" or end_from_max_trajectory or success or fail or (
                    self.machine_acts[-1] == "SUMMARIZE_AND_INFORM" and not success)
        else:
            done = end_from_max_trajectory or success or fail  # TODO REDO

        if self.t > 1000:
            logger.warning("[WARNING] dialogue size is {} with user {}".format(self.t, self.__user))

        if done:
            observation = None
        else:
            observation = State(self.t,
                                self.will_overflow_slots,
                                self.overflow_max_traj, reco,
                                self.reco_by_slot,
                                self.i_slot,
                                self.current_slot,
                                self.user_acts,
                                self.machine_acts,
                                {"errs": self.errs})._asdict()

        if success:
            logger.info("Dialogue is a sucess !")

        info = {"success": success,
                "state_is_absorbing": done,
                "fail": fail,
                "c_": 1. if u_action == "HANGUP" else 0.
                }

        return observation, reward, done, info

    def __gen_error_reco(self, err, cerr=-1, cok=1, std=0.2):

        reco_err = 1. / (1 + np.exp(-np.random.normal(cerr, std)))
        reco_sucess = 1. / (1 + np.exp(-np.random.normal(cok, std)))
        ret = reco_err if err else reco_sucess
        return ret.tolist()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    e = SlotFillingEnv(user_params={"cerr": -1, "cok": 1, "ser": 0.3, "cstd": 0.2, "proba_hangup": 0.3})
    e.seed(1)
    from ncarrara.bftq_pydial.tools.policies import HandcraftedSlotFillingEnv

    hdc_policy = HandcraftedSlotFillingEnv(safeness=0., system_actions=e.system_actions)

    for _ in range(10):
        hdc_policy.reset()
        s_ = e.reset()
        done = False
        while not done:
            s = s_
            a, _, _ = hdc_policy.execute(s)
            s_, r, done, info = e.step(a)
