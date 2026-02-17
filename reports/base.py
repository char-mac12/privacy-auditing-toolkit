from abc import ABC, abstractmethod
from attacks.attack_result import AttackResult

class BaseReporter(ABC):

    @abstractmethod
    def report(self, result: AttackResult):
        pass
