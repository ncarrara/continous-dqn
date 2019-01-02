import abc



class AbstractUser(object):
    ser = None
    current_action = None

    def __init__(self, ser, cerr, cok, cstd):
        self.cstd = cstd
        self.ser = ser
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
        pass

    @abc.abstractmethod
    def close(self):
        pass
