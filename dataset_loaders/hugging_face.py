from datasets import load_dataset

from core.registries import register, DATASET_REGISTRY
from dataset_loaders.base import BaseDataset
from core.logger import log, LogLevel

@register(DATASET_REGISTRY, "hugging-face")
class HuggingFaceDataset(BaseDataset):

    def __init__(self, config):
        self.dataset_name = config.get("dataset_path")
        self.split = config.get("split")

        log(f"[Dataset] Loading HF dataset: {self.dataset_name}", LogLevel.VERBOSE)

        dataset = load_dataset(self.dataset_name)

        if self.split:
            self.dataset = dataset[self.split]
        else:
            self.dataset = dataset