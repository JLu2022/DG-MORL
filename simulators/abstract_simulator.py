from abc import abstractmethod, ABC


class AbstractSimulator(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass
