from abc import ABC, abstractmethod

class Attack(ABC):
    display_name: str

    @abstractmethod
    def run(self, model, dataset):
        pass