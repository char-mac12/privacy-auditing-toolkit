from abc import ABC, abstractmethod
from typing import List

class BaseModel(ABC):
    display_name: str

    @abstractmethod
    def generate(self, input_data: str) -> str:
        pass

    @abstractmethod
    def loss(self, samples: List[str]) -> List[float]:
        pass