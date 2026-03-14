from datasets import load_dataset

from dataset_loaders.hugging_face import HuggingFaceDataset
from core.registries import register, DATASET_REGISTRY
from core.logger import log, LogLevel

@register(DATASET_REGISTRY, "wikimia")
class WikiMiaDataset(HuggingFaceDataset):
    def __init__(self, config):
        super().__init__(config)

        self.display_name = "WikiMIA"

        members = self.dataset.filter(lambda x: x["label"] == 1)
        non_members = self.dataset.filter(lambda x: x["label"] == 0)

        self._members = members["input"]
        self._non_members = non_members["input"]

        log(f"[Dataset] Loading {self.display_name}", LogLevel.VERBOSE)

    def member_samples(self):
        return self._members
    
    def non_member_samples(self):
        return self._non_members