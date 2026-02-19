from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from core.registries import register, METRICS_REGISTRY

@register(METRICS_REGISTRY, "loss-based-mia")
class MembershipInferenceMetrics:
    def compute(self, attack_output):
        member_losses = attack_output["member_losses"]
        non_member_losses = attack_output["non_member_losses"]

        y_true = [1]*len(member_losses) + [0]*len(non_member_losses)
        y_scores = [-l for l in member_losses + non_member_losses]

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        best_threshold_index = (tpr - fpr).argmax()
        threshold = thresholds[best_threshold_index]

        y_pred = [1 if score > threshold else 0 for score in y_scores]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
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