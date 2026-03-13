from pathlib import Path
from transformers import GPTNeoXForCausalLM, AutoTokenizer

from model_interfaces.causal_lm import CausalLmModel
from core.registries import register, MODEL_REGISTRY

@register(MODEL_REGISTRY, "llama")
class LlamaModel(CausalLmModel):
    
    def __init__(self, config=None):
        super().__init__(config)

        self.display_name = "Llama"
        
        model_path = Path(self.name)
        if model_path.exists():
            model_path = str(model_path.resolve())
        else:
            model_path = self.name
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPTNeoXForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()