from datasets import load_dataset

from dataset_loaders.base import BaseDataset
from core.registries import register, DATASET_REGISTRY
from core.logger import log, LogLevel

@register(DATASET_REGISTRY, "wikitext")
class WikiTextDataset(BaseDataset):
    
    def __init__(self, config):
        subset = config.get("subset", "wikitext-103-raw-v1")
        max_member = config.get("max_member_samples", 500)
        max_non_member = config.get("max_non_member_samples", 500)
        
        self.display_name = f"wikitext: {subset}"
        
        log(f"[Dataset] Loading {self.display_name}", LogLevel.INFO)
        
        train_data = load_dataset("wikitext", subset, split="train")
        val_data = load_dataset("wikitext", subset, split="validation")
        
        self._members = []
        for item in train_data:
            text = item.get("text", "").strip()
            if text and len(text) > 50:
                self._members.append(text)
            if len(self._members) >= max_member:
                break
        
        self._non_members = []
        for item in val_data:
            text = item.get("text", "").strip()
            if text and len(text) > 50:
                self._non_members.append(text)
            if len(self._non_members) >= max_non_member:
                break
        
        log(f"[Dataset] Loaded {len(self._members)} members, {len(self._non_members)} non-members", LogLevel.INFO)
    
    def member_samples(self):
        return self._members
    
    def non_member_samples(self):
        return self._non_members