from core.registries import register, ATTACK_REGISTRY
from attack_executor.base import BaseAttack
from core.logger import log, LogLevel

@register(ATTACK_REGISTRY, "loss-based-mia")
class LossBasedMIA(BaseAttack):
    display_name = "Loss-based MIA"
    higher_is_member = False

    def __init__(self, config=None):
        cfg = config or {}
        log(f"[Attack] Initialising {self.display_name}", LogLevel.VERBOSE)

    def score(self, model, samples):
        return model.loss(samples)