import json
from datetime import datetime

from attacks.attack_result import AttackResult
from core.config import LogLevel
from core.logger import log
from core.registries import REPORTER_REGISTRY, register
from reports.base import BaseReporter
from pathlib import Path

@register(REPORTER_REGISTRY, "json")
class JsonReporter(BaseReporter):
    def __init__(self, config=None):
        config = config or {}
        self.output_dir = Path(config.get("output_dir", "outputs/json"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log(f"[Reporter] JsonReporter initialized, saving to {self.output_dir}", LogLevel.VERBOSE)

    def report(self, result: AttackResult, run_config=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"Privacy_Audit_Report_{timestamp}.json"

        report_data = {
            "timestamp": timestamp,
            "attack_name": result.attack_name,
            "model_name": result.model_name,
            "dataset_name": result.dataset_name,
            "attack_duration": result.attack_duration,
            "run_config": run_config,
            "metrics": result.metrics,
            "attack_outputs": result.attack_outputs
        }

        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=4, default=str)