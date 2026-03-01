import torch
from tqdm import tqdm

from models.base import BaseModel

class CausalLmModel(BaseModel):
    
    def __init__(self, config=None):
        cfg = config or {}
        self.name = cfg.get("model_id", "gpt2")   # should this have a default? i think not probs
        device_str = cfg.get("device", "cuda")
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.max_seq_len = cfg.get("max_sequence_length", 512)
        self.batch_size = cfg.get("batch_size", 8)

        self.model = None
        self.tokenizer = None

    def loss(self, samples):
        losses = []

        for i in tqdm(range(0, len(samples), self.batch_size), desc="Calculating loss"):
            batch = samples[i:i + self.batch_size]

            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_len).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()

                loss_function = torch.nn.CrossEntropyLoss(reduction="none")
                token_losses = loss_function(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                token_losses = token_losses.view(len(batch), -1)
                attention_mask = inputs["attention_mask"][..., 1:].float()

                sample_losses = (token_losses * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
                losses.extend(sample_losses.cpu().tolist())

        return losses
        
    def generate(self, text, max_new_tokens=50):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    def per_token_loss(self, samples):
        per_token_losses_list = []
    
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i:i + self.batch_size]
            
            inputs = self.tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=self.max_seq_len
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                logits = outputs.logits
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()
                
                loss_function = torch.nn.CrossEntropyLoss(reduction="none")
                token_losses = loss_function(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                token_losses = token_losses.view(len(batch), -1)
                attention_mask = inputs["attention_mask"][..., 1:].float()
                
                # Extract per-token losses for each sample (excluding padding)
                for j in range(len(batch)):
                    # Get losses for this sample, masking out padding
                    sample_token_losses = token_losses[j]
                    sample_mask = attention_mask[j]
                    
                    # Only keep non-padding tokens
                    valid_losses = sample_token_losses[sample_mask == 1].cpu().tolist()
                    per_token_losses_list.append(valid_losses)
        
        return per_token_losses_list