from abc import ABC, abstractmethod

class InputHandler(ABC):

    @abstractmethod
    def _handle(self,*args):
        pass

    def handle(self, *args):
        return self._handle(*args)