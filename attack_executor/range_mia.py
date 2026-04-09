import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

from core.registries import register, ATTACK_REGISTRY
from attack_executor.base import BaseAttack
from core.logger import log, LogLevel
from attack_executor.range_sample_word_replace import sample_word_replace

@register(ATTACK_REGISTRY, "range-mia")
class RangeMIA(BaseAttack):
    """Membership inference attack based on scores of perturbed neighbour samples."""

    display_name = "Range MIA"

    def __init__(self, config=None):
        """Initialise the attack and configure neighbourhood sampling."""
        cfg = config or {}
        log(f"[Attack] {self.display_name} initialised", LogLevel.VERBOSE)

        self.base_attack_id = cfg.get("base_attack", "loss-based-mia")
        self.sample_size = cfg.get("sample_size", 10)

        # neighbourhood sample config
        self.num_masks = cfg.get("num_masks", 5)
        self.mask_model = cfg.get("mask_model", "bert-base-uncased")
        self.top_k = cfg.get("top_k", 6)
        device_config = cfg.get("device", "cuda")
        self.device = torch.device(device_config if torch.cuda.is_available() else "cpu")
        self.seed = cfg.get("seed", None)

        self.mlm_model = AutoModelForMaskedLM.from_pretrained(self.mask_model).to(self.device)
        self.mlm_tokenizer = AutoTokenizer.from_pretrained(self.mask_model)
        if self.mlm_tokenizer.pad_token is None:
            self.mlm_tokenizer.pad_token = self.mlm_tokenizer.eos_token

        # base attack config
        base_attack_cls = ATTACK_REGISTRY[self.base_attack_id]
        self.higher_is_member = base_attack_cls.higher_is_member

        # trimmed average config
        self.trim_start = cfg.get("trim_start", 0.0)
        self.trim_end = cfg.get("trim_end", 0.8)
        
    def score(self, model, samples):
        """Score samples using neighbour perturbations and a base attack."""
        scores = []

        base_attack_cls = ATTACK_REGISTRY[self.base_attack_id]
        base_attack = base_attack_cls()

        for idx, sample in enumerate(samples):
            sample_seed = self.seed + idx if self.seed is not None else None

            neighbourhood = self._generate_neighbourhood(sample, sample_seed)

            neighbour_scores = base_attack.score(model, neighbourhood)
            
            range_score = self._trimmed_average(neighbour_scores)
            scores.append(range_score)

        return scores

    def _generate_neighbourhood(self, sample, sample_seed):
        """Generate neighbouring samples using masked word replacement."""
        return sample_word_replace(
                sample, 
                self.mlm_model, 
                self.mlm_tokenizer, 
                self.num_masks, 
                self.sample_size, 
                self.top_k,
                self.device,
                sample_seed
            )
    
    def _trimmed_average(self, scores):
        """Compute the trimmed mean of neighbour scores to reduce outliers."""

        scores = np.array(scores)
        scores = np.sort(scores)

        n = len(scores)
        start_idx = int(np.ceil(self.trim_start * n))
        end_idx = int(np.floor(self.trim_end * n))

        trimmed_scores = scores[start_idx:end_idx]

        return float(np.mean(trimmed_scores))