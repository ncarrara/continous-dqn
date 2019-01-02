import abc

from env.slot_filling_2.env_slot_filling import U_NONE


class AbstractUser(object):
    ser = None
    current_action = None

    def __init__(self, ser, cerr, cok, cstd):
        self.cstd = cstd
        self.ser = ser
        self.current_action = U_NONE
        self.cerr = cerr
        self.cok = cok
        pass

    @abc.abstractmethod
    def action(self):
        pass  # pragma: no cover

    @abc.abstractmethod
    def update(self, user_observation):
        pass  # pragma: no cover

    @abc.abstractmethod
    def reset(self):
        self.current_action = U_NONE
        pass

    @abc.abstractmethod
    def close(self):
        pass
