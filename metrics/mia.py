from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from core.registries import register, METRICS_REGISTRY


def _compute_mia_metrics(member_values, non_member_values, higher_is_member=False):
    y_true = [1] * len(member_values) + [0] * len(non_member_values)
    
    if higher_is_member:
        y_scores = member_values + non_member_values
    else:
        y_scores = [-v for v in member_values + non_member_values]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    best_threshold_index = (tpr - fpr).argmax()
    threshold = thresholds[best_threshold_index]
    
    y_pred = [1 if score > threshold else 0 for score in y_scores]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_scores)
    advantage = recall - fpr[best_threshold_index]
    
    tpr_at_0_001_percent_fpr = tpr[fpr <= 0.00001][-1] if any(fpr <= 0.00001) else 0.0
    tpr_at_0_1_percent_fpr = tpr[fpr <= 0.001][-1] if any(fpr <= 0.001) else 0.0
    tpr_at_1_percent_fpr = tpr[fpr <= 0.01][-1] if any(fpr <= 0.01) else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "advantage": advantage,
        "threshold": threshold,
        "tpr_at_0.001%_fpr": tpr_at_0_001_percent_fpr,
        "tpr_at_0.1%_fpr": tpr_at_0_1_percent_fpr,
        "tpr_at_1%_fpr": tpr_at_1_percent_fpr
    }


@register(METRICS_REGISTRY, "loss-based-mia")
class LossBasedMiaMetrics:    

    def compute(self, attack_output):
        return _compute_mia_metrics(
            attack_output["member_losses"],
            attack_output["non_member_losses"],
            higher_is_member=False  # Lower loss = more likely member
        )


@register(METRICS_REGISTRY, "min-k-mia")
class MinKProbabilityMetrics:
    
    def compute(self, attack_output):
        return _compute_mia_metrics(
            attack_output["member_scores"],
            attack_output["non_member_scores"],
            higher_is_member=True  # Higher Min-K% score = more likely member
        )