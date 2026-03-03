from datetime import datetime
from pathlib import Path
import math
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay

from attacks.attack_result import AttackResult
from core.registries import register, REPORTER_REGISTRY
from core.logger import log, LogLevel
from core.run_config import RunConfig
from reports.base import BaseReporter

styles = getSampleStyleSheet()

@register(REPORTER_REGISTRY, "pdf")
class PdfReporter(BaseReporter):
    """Reporter that generates comprehensive PDF reports."""

    def __init__(self, config=None):
        config = config or {}
        self.output_dir = Path(config.get("output_dir", "outputs/pdf"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log(f"[Reporter] PDFReporter initialized, saving to {self.output_dir}", LogLevel.VERBOSE)

    def report(self, result: AttackResult, run_config=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        display_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        filepath = self.output_dir / f"Privacy_Audit_Report_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        story = []

        story.append(Spacer(1, 0.5*inch))

        # Audit Summary
        story.append(Paragraph("Audit Summary", styles["Heading2"]))
        story.append(Spacer(1, 0.1*inch))
        story.append(self._summary_table(result, display_time))
        story.append(Spacer(1, 0.3*inch))

        # Attack Details

        # Model Details
        if run_config:
            story.append(Paragraph("Model Details", styles["Heading2"]))
            story.append(Spacer(1, 0.1*inch))
            story.append(self._model_config_table(result, run_config))
            story.append(Spacer(1, 0.3*inch))

        # Dataset Details
        story.append(Paragraph("Dataset Details", styles["Heading2"]))
        story.append(Spacer(1, 0.1*inch))
        story.append(self._dataset_table(result))
        story.append(Spacer(1, 0.3*inch))

        # Metrics
        story.append(Paragraph("Attack Performance Metrics", styles["Heading2"]))
        story.append(Spacer(1, 0.1*inch))
        story.append(self._metrics_table(result))
        story.append(Spacer(1, 0.3*inch))

        roc_path = self._roc_curve(result)
        if roc_path:
            story.append(Paragraph("ROC Curve", styles["Heading2"]))
            story.append(Spacer(1, 0.1*inch))
            story.append(Image(str(roc_path), width=5*inch, height=4*inch))

        loss_path = self._loss_distribution(result)
        if loss_path:
            story.append(Paragraph("Loss Distribution", styles["Heading2"]))
            story.append(Spacer(1, 0.1*inch))
            story.append(Image(str(loss_path), width=5*inch, height=4*inch))

        if result.attack_name != "Range MIA":
            cm_path = self._confusion_matrix_plot(result)
            if cm_path:
                story.append(Paragraph("Confusion Matrix", styles["Heading2"]))
                story.append(Spacer(1, 0.1*inch))
                story.append(Image(str(cm_path), width=5*inch, height=4*inch))

        doc.build(story, onFirstPage=self._first_page, onLaterPages=self._later_pages)

    def _first_page(self, canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica-Bold', 16)
        canvas.drawCentredString(letter[0]/2, letter[1]-108, "Privacy Audit Report")
        canvas.setFont('Helvetica', 9)
        canvas.drawString(inch, 0.75*inch, "Page 1")
        canvas.restoreState()

    def _later_pages(self, canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        canvas.drawString(inch, 0.75*inch, f"Page {doc.page}")
        canvas.restoreState()

    def _summary_table(self, result: AttackResult, timestamp: str):
        data = [
            ["Attack", result.attack_name],
            ["Model", result.model_name],
            ["Dataset", result.dataset_name],
            ["Report Creation", timestamp],
            ["Attack Duration", result.attack_duration]
        ]
        
        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        return table
    
    def _model_config_table(self, result: AttackResult, run_config: RunConfig):
        model_config = run_config.model_config
        
        data = []
        
        # Add relevant config items
        if "model_id" in model_config:
            data.append(["Model ID", model_config["model_id"]])
        # if "device" in model_config:
        #     data.append(["Device", model_config["device"]])
        if "max_sequence_length" in model_config:
            data.append(["Max Sequence Length", str(model_config["max_sequence_length"])])
        if "batch_size" in model_config:
            data.append(["Batch Size", str(model_config["batch_size"])])

        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        return table

    def _dataset_table(self, result: AttackResult):
        member_losses = result.attack_outputs.get("member_losses", [])
        member_scores = result.attack_outputs.get("member_scores", [])
        num_member_samples = len(member_losses) if member_losses else len(member_scores)
        
        non_member_losses = result.attack_outputs.get("non_member_losses", [])
        non_member_scores = result.attack_outputs.get("non_member_scores", [])
        num_non_member_samples = len(non_member_losses) if non_member_losses else len(non_member_scores)

        g = math.gcd(num_member_samples, num_non_member_samples)

        simplified_a = num_member_samples // g
        simplified_b = num_non_member_samples // g

        formatted_ratio = f"{simplified_a}:{simplified_b}"

        data = [
            ["Dataset", result.dataset_name],
            ["Member samples", str(num_member_samples)],
            ["Non-member samples", str(num_non_member_samples)],
            ["Total samples", str(num_member_samples + num_non_member_samples)],
            ["Member/Non-member ratio", formatted_ratio]
        ]

        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        return table

    def _metrics_table(self, result: AttackResult):
        if not result.metrics:
            return Paragraph("No metrics available", styles["Normal"])
        
        data = [["Metric", "Value"]]
        for metric, value in result.metrics.items():
            display_name = metric.replace("_", " ").title()
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            data.append([display_name, formatted_value])
        
        table = Table(data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        return table
    
    def _roc_curve(self, result: AttackResult):
        attack_outputs = result.attack_outputs or {}
        member_scores = attack_outputs.get("member_scores")
        non_member_scores = attack_outputs.get("non_member_scores")
        higher_is_member = attack_outputs.get("higher_is_member")

        member_scores = [v for v in member_scores if np.isfinite(v)]
        non_member_scores = [v for v in non_member_scores if np.isfinite(v)]
    
        if len(member_scores) == 0 or len(non_member_scores) == 0:
            raise ValueError("No valid scores available to compute metrics")
        
        y_true = [1] * len(member_scores) + [0] * len(non_member_scores)

        if higher_is_member:
            y_scores = (member_scores + non_member_scores)
        else:
            y_scores = [-score for score in member_scores + non_member_scores]
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plot_path = self.output_dir / f"roc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _loss_distribution(self, result: AttackResult):
        attack_outputs = result.attack_outputs or {}
        member_losses = attack_outputs.get("member_losses")
        non_member_losses = attack_outputs.get("non_member_losses")
        
        if not member_losses or not non_member_losses:
            return None
        
        plt.figure(figsize=(6, 5))
        plt.hist(
        member_losses,
        bins=50,
        alpha=0.6,
        label="Members",
        density=True
        )
        plt.hist(
            non_member_losses,
            bins=50,
            alpha=0.6,
            label="Non-members",
            density=True
        )

        plt.xlabel("Loss")
        plt.ylabel("Density")
        plt.title("Loss Distribution")
        plt.legend()
        plt.grid(alpha=0.3)

        plot_path = self.output_dir / f"loss_dist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        return plot_path
    
    def _confusion_matrix_plot(self, result: AttackResult):
        attack_outputs = result.attack_outputs or {}
        member_losses = attack_outputs.get("member_losses")
        non_member_losses = attack_outputs.get("non_member_losses")
        threshold = result.metrics.get("threshold")

        if not member_losses or not non_member_losses or threshold is None:
            return None

        y_true = [1] * len(member_losses) + [0] * len(non_member_losses)
        y_scores = [-l for l in member_losses + non_member_losses]
        y_pred = [1 if s >= threshold else 0 for s in y_scores]

        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            display_labels=["Non-Member", "Member"],
            cmap="Blues",
            values_format="d"
        )

        plt.title("Confusion Matrix")
        plt.grid(False)

        plot_path = self.output_dir / f"cm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        return plot_path