import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from attack_executor.range_samplers import sample_word_replace  # Import the function


class RangeSampler:
    def __init__(self, range_fn="word_replace", sample_size=10, config=None):
        self.range_fn = range_fn
        self.sample_size = sample_size
        self.config = config or {}

        self.num_masks = self.config.get("num_masks", 1)
        self.mlm_model_name = self.config.get("mask_model", "bert-base-uncased")
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        if self.range_fn == "word_replace":
            self._mlm_model = AutoModelForMaskedLM.from_pretrained(self.mlm_model_name).to(self.device)
            self._mlm_tokenizer = AutoTokenizer.from_pretrained(self.mlm_model_name)
            if self._mlm_tokenizer.pad_token is None:
                self._mlm_tokenizer.pad_token = self._mlm_tokenizer.eos_token

    def sample(self, text):
        if self.range_fn == "word_replace":
            return sample_word_replace(  # Use the paper's function
                text,
                self._mlm_model,
                self._mlm_tokenizer,
                self.num_masks,
                self.sample_size,
                self.device
            )
        else:
            raise NotImplementedError(f"Range function {self.range_fn} not implemented yet.")