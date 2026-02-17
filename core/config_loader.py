import json

from core.logger import log, LogLevel
from core.run_config import RunConfig

class ConfigLoader:
    @staticmethod
    def load(filepath: str) -> RunConfig:
        log(f"[ConfigLoader] Loading JSON config from: {filepath}", LogLevel.INFO)

        with open(filepath, "r") as f:
            cfg = json.load(f)
        
        log(f"[ConfigLoader] Raw JSON content: {cfg}", LogLevel.VERBOSE)

        run_config = RunConfig(
            model_id=cfg["model"]["id"],
            dataset_id=cfg["dataset"]["id"],
            attack_id=cfg["attack"]["id"],
            reporter_id=cfg["reporter"]["id"],

            model_config=cfg["model"],
            dataset_config=cfg["dataset"],
            attack_config=cfg.get("attack", {}),
            reporter_config=cfg.get("reporter", {})
        )

        return run_config