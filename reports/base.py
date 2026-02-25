from abc import ABC, abstractmethod
from attacks.attack_result import AttackResult
from core.run_config import RunConfig

class BaseReporter(ABC):

    @abstractmethod
    def report(self, result: AttackResult, run_config: RunConfig = None):
        pass
