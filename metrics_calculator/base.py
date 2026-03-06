from abc import ABC, abstractmethod
from typing import List

from attack_executor.attack_result import AttackResult

class BaseMetrics(ABC):

    @abstractmethod
    def compute(self, attack_output: AttackResult) -> dict:
        pass