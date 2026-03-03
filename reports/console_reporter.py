from datetime import datetime

from attacks.attack_result import AttackResult
from core.config import LogLevel
from core.logger import log
from core.registries import REPORTER_REGISTRY, register
from reports.base import BaseReporter


@register(REPORTER_REGISTRY, "console")
class PdfReporter(BaseReporter):
    def __init__(self, config=None):
        config = config or {}
        log(f"[Reporter] Console reporter initialised", LogLevel.VERBOSE)

    def report(self, result: AttackResult, run_config=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        display_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")

        print("---------------------------------------------------------------")
        print(f"Audit completed at {display_time}")
        print("---------------------------------------------------------------")
        print(f"Attack: {result.attack_name}")
        print(f"Model: {result.model_name}")
        print(f"Dataset: {result.dataset_name}")
        print(f"Attack duration: {result.attack_outputs.get("attack_duration")}")
        print("---------------------------------------------------------------")
        for metric, value in result.metrics.items():
            metric_display_name = metric.replace("_", " ").title()
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            print(f"{metric_display_name}: {formatted_value}")