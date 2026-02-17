from core.registries import register, ATTACK_REGISTRY
from base import Attack
from core.logger import log, LogLevel
from attack_result import AttackResult


@register(ATTACK_REGISTRY, "loss-based-mia")
class LossBasedMIA(Attack):
    display_name = "Loss-based Membership Inference Attack"

    def run(self, model, dataset):
        log(
            f"[Attack] Running {self.display_name} "
            f"on model '{model.name}' with dataset '{dataset.name}'",
            LogLevel.INFO
        )

        member_losses = model.loss(dataset.member_samples())
        non_member_losses = model.loss(dataset.non_member_samples())

        log("[Attack] Loss-based MIA finished", LogLevel.INFO)

        return AttackResult(
            attack_name=self.display_name,
            model_name=model.name,
            dataset_name=dataset.name,
            attack_outputs={
                "member_losses": member_losses,
                "non_member_losses": non_member_losses,
            },
            summary=(f"{self.display_name} completed on {len(member_losses) + len(non_member_losses)} samples")
        )
