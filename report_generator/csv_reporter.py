import csv
import json
from datetime import datetime

from attack_executor.attack_result import AttackResult
from core.config import LogLevel
from core.logger import log
from core.registries import REPORTER_REGISTRY, register
from report_generator.base import BaseReporter
from pathlib import Path

@register(REPORTER_REGISTRY, "csv")
class CsvReporter(BaseReporter):
    def __init__(self, config=None):
        config = config or {}
        self.output_dir = Path(config.get("output_dir", "outputs/csv"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log(f"[Reporter] CsvReporter initialized, saving to {self.output_dir}", LogLevel.VERBOSE)

    def report(self, result: AttackResult, run_config=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"Privacy_Audit_Report_{timestamp}.csv"

        report_data = {
            "timestamp": timestamp,
            "attack_name": result.attack_name,
            "model_name": result.model_name,
            "dataset_name": result.dataset_name,
            "attack_duration": result.attack_duration
        }

        report_data.update(result.metrics)
        
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=report_data.keys())
            writer.writeheader()
            writer.writerow(report_data)