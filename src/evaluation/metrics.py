"""Evaluation metrics and protocol for binary deepfake detection."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import torch


@dataclass
class EvaluationMetrics:
    """Standard container for deepfake classification and verification metrics."""

    loss: Optional[float]
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    eer: float
    confusion_matrix: Dict[str, int]  # "tp", "fp", "tn", "fn"
    num_samples: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to dictionary."""
        return asdict(self)

    def summary_string(self) -> str:
        """Human-readable metric summary string."""
        loss_str = f"{self.loss:.4f}" if self.loss is not None else "N/A"
        auc_str = f"{self.roc_auc:.4f}" if not np.isnan(self.roc_auc) else "NaN (single-class)"
        eer_str = f"{self.eer:.4f}" if not np.isnan(self.eer) else "NaN (single-class)"
        return (
            f"Samples: {self.num_samples} | Loss: {loss_str} | "
            f"Acc: {self.accuracy:.4f} | Prec: {self.precision:.4f} | "
            f"Rec: {self.recall:.4f} | F1: {self.f1:.4f} | "
            f"AUC: {auc_str} | EER: {eer_str} | "
            f"CM: [TP={self.confusion_matrix['tp']}, FP={self.confusion_matrix['fp']}, "
            f"TN={self.confusion_matrix['tn']}, FN={self.confusion_matrix['fn']}]"
        )


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Equal Error Rate (EER) via ROC curve and linear crossing interpolation.

    Mathematical formulation:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fnr = 1.0 - tpr
        d = fpr - fnr

    Search for adjacent ROC points i and i+1 where d[i] * d[i+1] <= 0 (crossing).
    If bracketed:
        w = -d[i] / (d[i+1] - d[i])
        eer = fpr[i] + w * (fpr[i+1] - fpr[i])
    If no sign change exists across all points:
        eer = (fpr[argmin(|d|)] + fnr[argmin(|d|)]) / 2.0
    Clamped strictly to [0.0, 1.0].
    If len(np.unique(y_true)) < 2: returns float('nan').

    Args:
        y_true: Binary ground truth array {0, 1}.
        y_score: Continuous prediction scores / probabilities.

    Returns:
        Equal Error Rate in [0.0, 1.0], or NaN if single-class.
    """
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    d = fpr - fnr

    # Search for adjacent points bracketing a zero-crossing
    crossing_found = False
    eer = 0.5

    for i in range(len(d) - 1):
        if d[i] * d[i + 1] <= 0:
            denom = d[i + 1] - d[i]
            if abs(denom) > 1e-12:
                w = -d[i] / denom
                fpr_interp = fpr[i] + w * (fpr[i + 1] - fpr[i])
                fnr_interp = fnr[i] + w * (fnr[i + 1] - fnr[i])
                eer = (fpr_interp + fnr_interp) / 2.0
            else:
                eer = (fpr[i] + fnr[i]) / 2.0
            crossing_found = True
            break

    if not crossing_found:
        best_idx = int(np.argmin(np.abs(d)))
        eer = (fpr[best_idx] + fnr[best_idx]) / 2.0

    return float(np.clip(eer, 0.0, 1.0))


def compute_metrics(
    y_true: Union[torch.Tensor, np.ndarray],
    y_logits: Union[torch.Tensor, np.ndarray],
    loss: Optional[float] = None,
    threshold: float = 0.5,
) -> EvaluationMetrics:
    """Compute complete classification and ranking metrics from logits and targets.

    Invariants:
        - y_true and y_logits must have identical shapes of strictly (N, 1).
        - Probabilities are calculated as torch.sigmoid(logits).
        - Single-class inputs produce NaN for ROC-AUC and EER without throwing errors.
        - Empty inputs (0 samples) raise explicit ValueError.

    Args:
        y_true: Ground truth binary targets of shape (N, 1).
        y_logits: Raw unnormalized scalar model logits of shape (N, 1).
        loss: Optional scalar loss value for the evaluated split.
        threshold: Classification decision threshold (default: 0.5).

    Returns:
        Populated EvaluationMetrics dataclass instance.
    """
    if isinstance(y_true, np.ndarray):
        y_true_t = torch.from_numpy(y_true)
    elif isinstance(y_true, torch.Tensor):
        y_true_t = y_true
    else:
        raise TypeError(f"y_true must be torch.Tensor or np.ndarray, got {type(y_true)}")

    if isinstance(y_logits, np.ndarray):
        y_logits_t = torch.from_numpy(y_logits)
    elif isinstance(y_logits, torch.Tensor):
        y_logits_t = y_logits
    else:
        raise TypeError(f"y_logits must be torch.Tensor or np.ndarray, got {type(y_logits)}")

    if y_true_t.shape != y_logits_t.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape {tuple(y_true_t.shape)} != y_logits shape {tuple(y_logits_t.shape)}. "
            "Both must be strictly (N, 1)."
        )

    if y_true_t.dim() != 2 or y_true_t.shape[1] != 1:
        raise ValueError(
            f"Expected 2D tensors of shape (N, 1), got y_true shape {tuple(y_true_t.shape)} and "
            f"y_logits shape {tuple(y_logits_t.shape)}"
        )

    n_samples = y_true_t.shape[0]
    if n_samples == 0:
        raise ValueError("Cannot compute metrics on empty inputs (0 samples).")

    # Diagnostic probabilities
    probs_t = torch.sigmoid(y_logits_t)
    probs = probs_t.detach().cpu().numpy().flatten()
    y_true_np = y_true_t.detach().cpu().numpy().flatten().astype(int)
    y_pred_np = (probs >= threshold).astype(int)

    # Confusion matrix
    tp = int(np.sum((y_true_np == 1) & (y_pred_np == 1)))
    fp = int(np.sum((y_true_np == 0) & (y_pred_np == 1)))
    tn = int(np.sum((y_true_np == 0) & (y_pred_np == 0)))
    fn = int(np.sum((y_true_np == 1) & (y_pred_np == 0)))

    # Classification metrics
    accuracy = float((tp + tn) / n_samples)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # ROC-AUC & EER with single-class guard
    unique_labels = np.unique(y_true_np)
    if len(unique_labels) < 2:
        roc_auc = float("nan")
        eer = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true_np, probs))
        eer = compute_eer(y_true_np, probs)

    return EvaluationMetrics(
        loss=float(loss) if loss is not None else None,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        eer=eer,
        confusion_matrix={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        num_samples=n_samples,
    )
