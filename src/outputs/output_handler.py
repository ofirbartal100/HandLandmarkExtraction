from abc import ABC, abstractmethod

class OutputHandler(ABC):

    @abstractmethod
    def _handle(self,*args):
        pass

    def handle(self, *args):
        return self._handle(*args)