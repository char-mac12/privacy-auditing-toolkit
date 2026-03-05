import numpy as np

from core.registries import register, ATTACK_REGISTRY
from attacks.base import BaseAttack
from core.logger import log, LogLevel

@register(ATTACK_REGISTRY, "min-k-mia")
class MinKProbabilityMIA(BaseAttack):
    display_name = "Min-K% Probability MIA"
    higher_is_member = True
    
    def __init__(self, config=None):
        cfg = config or {}
        self.k_percent = cfg.get("k_percent", 20)
        log(f"Initialising {self.display_name}", LogLevel.VERBOSE)
    
    def score(self, model, samples):
        scores = []
    
        per_token_losses_list = model.per_token_loss(samples)
        
        for token_losses in per_token_losses_list:
            if len(token_losses) == 0:
                scores.append(0.0)
                continue
            
            # p = exp(log_p) = exp(-loss)
            token_probs = [np.exp(-loss) for loss in token_losses]
            
            # Sort in ascending order (lowest probs first)
            sorted_probs = sorted(token_probs)
            
            # Take bottom k%
            k = max(1, int(len(sorted_probs) * self.k_percent / 100))
            min_k_tokens = sorted_probs[:k]
            
            # Average of the k% hardest tokens
            # Higher average = higher probs on hard tokens = member
            min_k_score = np.mean(min_k_tokens)
            
            scores.append(min_k_score)
        
        return scores