from abc import ABC, abstractmethod
from utils.utils import describe


class Model(ABC):
    def __init__(self, args, max_time_steps):
        self.args = args
        self.maxTimeSteps = max_time_steps
        self.build_graph(args, max_time_steps)

    @describe
    @abstractmethod
    def build_graph(self, args, max_time_steps):
        pass
