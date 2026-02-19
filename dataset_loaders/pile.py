from datasets import load_dataset

from dataset_loaders.base import BaseDataset
from core.registries import register, DATASET_REGISTRY
from core.logger import log, LogLevel

@register(DATASET_REGISTRY, "pile")
class PileDataset(BaseDataset):
    """
    Config example:
    {
        "id": "pile",
        "subset": "pile_cc",  # or "wikipedia", "arxiv", "github", etc.
        "max_member_samples": 500,
        "max_non_member_samples": 500
    }
    """
    
    def __init__(self, config):
        self.subset = config.get("subset", "pile_cc")
        self.max_member = config.get("max_member_samples", 500)
        self.max_non_member = config.get("max_non_member_samples", 500)
        
        self.display_name = f"pile-{self.subset}"
        
        log(f"[Dataset] Loading The Pile subset: {self.subset}", LogLevel.VERBOSE)
        
        train_data = load_dataset(
            "EleutherAI/pile", 
            self.subset,
            split="train",
            streaming=True  
        )
        
        val_data = load_dataset(
            "EleutherAI/pile",
            self.subset, 
            split="validation",
            streaming=True
        )
        
        self._members = []
        for item in train_data:
            text = item.get("text", "").strip()
            if text and len(text) > 50:
                self._members.append(text)
            if len(self._members) >= self.max_member:
                break
        
        self._non_members = []
        for item in val_data:
            text = item.get("text", "").strip()
            if text and len(text) > 50:
                self._non_members.append(text)
            if len(self._non_members) >= self.max_non_member:
                break
        
        log(f"[Dataset] Loaded {len(self._members)} members, {len(self._non_members)} non-members", LogLevel.INFO)
    
    def member_samples(self):
        return self._members
    
    def non_member_samples(self):
        return self._non_members