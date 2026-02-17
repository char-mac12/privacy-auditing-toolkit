from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class AttackResult:
    attack_name: str
    model_name: str
    dataset_name: str
    attack_outputs: Dict[str, Any] | None = None
    metrics: Dict[str, float] | None = None
    summary: str | None = None