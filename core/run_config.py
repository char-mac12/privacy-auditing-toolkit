from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RunConfig:
    model_id: str
    dataset_id: str
    attack_id: str
    reporter_id: str

    model_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    attack_config: Dict[str, Any]
    reporter_config: Dict[str, Any]