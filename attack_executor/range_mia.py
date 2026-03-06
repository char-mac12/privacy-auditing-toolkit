import numpy as np

from core.registries import register, ATTACK_REGISTRY
from attack_executor.base import BaseAttack
from core.logger import log, LogLevel
from attack_executor.range_sampler import RangeSampler

@register(ATTACK_REGISTRY, "range-mia")
class RangeMIA(BaseAttack):
    display_name = "Range MIA"
    paraphrase_methods = ["word_replace", "prefix"]

    def __init__(self, config=None):
        cfg = config or {}
        log(f"")

        self.base_attack_id = cfg.get("base_attack", "loss-based-mia")
        self.sample_size = cfg.get("sample_size", 10)
        self.paraphrase_method = cfg.get("paraphrase_method", "word_replace")
        self.config = cfg

        base_attack_cls = ATTACK_REGISTRY[self.base_attack_id]
        self.higher_is_member = base_attack_cls.higher_is_member

        self._sampler = RangeSampler(range_fn=self.paraphrase_method,
                                     sample_size=self.sample_size,
                                     config=self.config)

        
    def score(self, model, samples):
        scores = []

        base_attack_cls = ATTACK_REGISTRY[self.base_attack_id]
        base_attack = base_attack_cls()

        for sample in samples:
            neighbourhood = self._generate_neighbourhood(sample)

            neighbour_scores = base_attack.score(model, neighbourhood)
            
            range_score = float(np.mean(neighbour_scores))
            scores.append(range_score)

        return scores
    
    def _generate_neighbourhood(self, sample):
        if self.paraphrase_method not in self.paraphrase_methods:
            raise NotImplementedError(
                f"Range MIA currently only supports {self.paraphrase_methods}. "
                f"Requested method: {self.paraphrase_method}."
            )

        return self._sampler.sample(sample)