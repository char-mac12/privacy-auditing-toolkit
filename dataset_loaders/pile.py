from datasets import load_from_disk

from dataset_loaders.base import BaseDataset
from core.registries import register, DATASET_REGISTRY
from core.logger import log, LogLevel

@register(DATASET_REGISTRY, "pile")
class PileDataset(BaseDataset):
    def __init__(self, config):
        self.display_name = "The Pile"

        self.dataset = load_from_disk(config.dataset_path)

        print(self.dataset)

        self._members = self.dataset["member"]
        self._non_members = self.dataset["nonmember"]

        log(f"[Dataset] Loading {self.display_name}", LogLevel.VERBOSE)

    def member_samples(self):
        return self._members
    
    def non_member_samples(self):
        return self._non_members