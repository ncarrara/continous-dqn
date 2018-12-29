from collections import namedtuple
import numpy as np

class Transition(namedtuple("Transition", ("s", "a", "r_", "s_"))):

    def numpy(self):
        return np.concatenate((self.s, self.a, self.r_, self.s_))

    def __str__(self):
        return "".join(["(", str(self.s), ", ", str(self.a), ", ", str(self.r_), ", ", str(self.s_)+ ")"])


class TransitionGym(namedtuple("TransitionGym", ("s", "a", "r_", "s_", "done", "info"))):

    def __str__(self):
        return "".join(["(", str(self.s), ", ", str(self.a), ", ", str(self.r_), ", ", str(self.s_),str(self.done), ", ", str(self.info)+ ")"])
