from core.logger import log, LogLevel
from abc import ABC, abstractmethod
from typing import List, Any

from attacks.attack_result import AttackResult
from datetime import datetime

class BaseAttack(ABC):
    display_name: str
    higher_is_member: bool    # if true, higher scores mean the attack should predict member, if false, lower scores = member

    def run(self, model, dataset):
        start_time = datetime.now()
        log(
            f"[Attack] Running {self.display_name} "
            f"on model '{model.display_name}' with dataset '{dataset.display_name}'",
            LogLevel.INFO
        )

        member_samples = dataset.member_samples()
        non_member_samples = dataset.non_member_samples()
        
        log("Calculating scores for members", LogLevel.VERBOSE)
        member_scores = self.score(model, member_samples)

        log("Calculating scores for non-members", LogLevel.VERBOSE)
        non_member_scores = self.score(model, non_member_samples)
        
        log(f"[Attack] {self.display_name} finished", LogLevel.INFO)

        end_time = datetime.now()
        attack_duration = end_time - start_time
        
        return AttackResult(
            attack_name=self.display_name,
            model_name=model.display_name,
            dataset_name=dataset.display_name,
            attack_duration=attack_duration,
            attack_outputs={
                "member_scores": member_scores,
                "non_member_scores": non_member_scores,
                "higher_is_member": self.higher_is_member
            },
            summary=(
                f"{self.display_name} completed on "
                f"{len(member_samples) + len(non_member_samples)} samples"
            )
        )

    @abstractmethod
    def score(self, model, samples) -> List[Any]:
        pass