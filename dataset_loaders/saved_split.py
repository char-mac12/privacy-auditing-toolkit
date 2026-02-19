"""
for pre-split member/non-member data saved in Json files
"""

import json
from pathlib import Path

from dataset_loaders.base import BaseDataset
from core.registries import register, DATASET_REGISTRY
from core.logger import log, LogLevel


@register(DATASET_REGISTRY, "saved-split")
class SavedSplitDataset(BaseDataset):
    """
    Load member/non-member samples from saved JSON files.
    
    Config example:
    {
        "id": "saved-split",
        "data_dir": "finetune/gpt2_wikitext_20240115/data_split",
        "member_file": "members.json",
        "non_member_file": "non_members.json"
    }
    """
    
    def __init__(self, config):
        self.data_dir = Path(config["data_dir"])
        self.member_file = config.get("member_file", "members.json")
        self.non_member_file = config.get("non_member_file", "non_members.json")

        self.display_name = "saved split"
        
        # Load metadata if available
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.display_name = f"{metadata.get('dataset', 'unknown')}-saved-split"
        else:
            self.display_name = "saved-split"
        
        # Load members
        member_path = self.data_dir / self.member_file
        if not member_path.exists():
            raise FileNotFoundError(f"Member samples file not found: {member_path}")
        
        with open(member_path, "r") as f:
            self._members = json.load(f)
        
        # Load non-members
        non_member_path = self.data_dir / self.non_member_file
        if not non_member_path.exists():
            raise FileNotFoundError(f"Non-member samples file not found: {non_member_path}")
        
        with open(non_member_path, "r") as f:
            self._non_members = json.load(f)
        
        log(f"[Dataset] Loaded dataset from {self.data_dir}", LogLevel.INFO)
        log(f"[Dataset] Dataset has {len(self._members)} members and {len(self._non_members)} non-members", LogLevel.VERBOSE)
    
    def member_samples(self):
        return self._members
    
    def non_member_samples(self):
        return self._non_members