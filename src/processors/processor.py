from abc import ABC, abstractmethod


class Processor(ABC):

    @abstractmethod
    def _process(self, input):
        pass

    def process(self, input):
        return self._process(input)
