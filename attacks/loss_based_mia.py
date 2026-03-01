from core.registries import register, ATTACK_REGISTRY
from attacks.base import Attack
from core.logger import log, LogLevel
from attacks.attack_result import AttackResult


@register(ATTACK_REGISTRY, "loss-based-mia")
class LossBasedMIA(Attack):
    display_name = "Loss-based Membership Inference Attack"
    higher_is_member = False

    def score(self, model, samples):
        return model.loss(samples)