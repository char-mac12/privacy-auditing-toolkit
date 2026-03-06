from abc import ABC, abstractmethod
from attack_executor.attack_result import AttackResult
from core.run_config import RunConfig

class BaseReporter(ABC):

    @abstractmethod
    def report(self, result: AttackResult, run_config: RunConfig = None):
        pass
