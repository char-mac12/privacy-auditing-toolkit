from abc import ABC, abstractmethod
from typing import List, Any

class BaseDataset(ABC):
    display_name: str

    @abstractmethod
    def member_samples(self) -> List[Any]:
        pass

    @abstractmethod
    def non_member_samples(self) -> List[Any]:
        pass