from core.run_config import RunConfig
from core.logger import log, LogLevel
from core.registries import MODEL_REGISTRY, DATASET_REGISTRY, ATTACK_REGISTRY, METRICS_REGISTRY, REPORTER_REGISTRY

class AttackRunner:
    def __init__(self, config: RunConfig):
        self.config = config

    def run(self):
        log("[Runner] Starting run", LogLevel.INFO)

        model_cls = MODEL_REGISTRY[self.config.model_id]
        model = model_cls(self.config.model_config)

        dataset_cls = DATASET_REGISTRY[self.config.dataset_id]
        dataset = dataset_cls(self.config.dataset_config)

        log(ATTACK_REGISTRY.keys(), LogLevel.INFO)
        attack_cls = ATTACK_REGISTRY[self.config.attack_id]
        attack_config = self.config.attack_config
        attack = attack_cls(attack_config)

        result = attack.run(model, dataset)

        if self.config.attack_id in METRICS_REGISTRY:
            metric_cls = METRICS_REGISTRY[self.config.attack_id]
            metrics = metric_cls().compute(result.attack_outputs)
            result.metrics = metrics

        reporter_cls = REPORTER_REGISTRY[self.config.reporter_id]
        reporter = reporter_cls(self.config.reporter_config)
        reporter.report(result, self.config)

        log("[Runner] Run complete", LogLevel.INFO)
