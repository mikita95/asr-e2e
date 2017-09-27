from abc import ABC, abstractmethod
from utils.utils import describe


class Model(ABC):
    def __init__(self, args, max_time_steps, num_classes):
        self.args = args
        self.maxTimeSteps = max_time_steps
        self.num_classes = num_classes
        self.build_graph()

    @describe
    @abstractmethod
    def build_graph(self):
        pass
