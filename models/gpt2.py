from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from models.causal_lm import CausalLmModel
from core.registries import register, MODEL_REGISTRY

@register(MODEL_REGISTRY, "gpt2")
class Gpt2Model(CausalLmModel):

    def __init__(self, config=None):
        super().__init__(config)

        model_path = Path(self.name)
        if model_path.exists():
            model_path = str(model_path.resolve())
        else:
            model_path = self.name

        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.model.eval()