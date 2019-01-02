from env.slot_filling_2.user.abstract_user import AbstractUser
from env.slot_filling_2.env_slot_filling import *

import logging


class HandcraftedUser(AbstractUser):

    def __init__(self, ser, cerr, cok, cstd, proba_handgup
                 # option_hangup_repeat={"decay": None, "number": None},
                 ):
        super(HandcraftedUser, self).__init__(ser, cerr, cok, cstd)
        self.proba_hangup = proba_handgup
        # self.repeat_proba = self.exp_decay(self.decay_repeat_oral)

    # def exp_decay(self, decay):
    #     return lambda x: 1. - np.exp(-x / (1. / decay))

    def reset(self):
        super(HandcraftedUser, self).reset()

        # self.nb_repeat_oral = 0

    # def repeat_oral_pissed_of(self):
    #     b0 = False
    #     b1 = False
    #     if self.decay_repeat_oral is not None:
    #         b0 = np.random.rand() < self.repeat_proba(self.nb_repeat_oral)
    #     if self.number_repeat_oral_max is not None:
    #         b1 = self.nb_repeat_oral >= self.number_repeat_oral_max
    #     return b0 or b1

    def repeat_num_pad_pissed_of(self):
        return self.proba_hangup is not None and np.random.rand() < self.proba_hangup

    def action(self, observation):
        other = {"user_is_pissed_of_by_repeat": False,
                 "user_is_pissed_of_by_implconf": False}
        machine_act = observation.machine_acts[-1]
        if np.random.rand() < 0.0:
            action = HANGUP
        elif machine_act == ASK_NEXT:
            action = INFORM_CURRENT
        elif machine_act == REPEAT_ORAL:
            if self.repeat_oral_pissed_of():
                action = HANGUP
            else:
                action = INFORM_CURRENT
                self.nb_repeat_oral += 1
        elif machine_act == REPEAT_NUMERIC_PAD:
            if self.repeat_num_pad_pissed_of():
                action = HANGUP
            else:
                action = INFORM_CURRENT
        else:
            # print machine_act
            raise Exception("oups pas possible")
        self.current_action = action
        return action, other

    def close(self):
        pass
