"""Probability calibration, temperature scaling, and optimal threshold search utilities."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss


@dataclass
class CalibrationMetrics:
    """Container for calibration assessment metrics."""

    ece: float  # Expected Calibration Error in [0.0, 1.0]
    mce: float  # Maximum Calibration Error in [0.0, 1.0]
    nll: float  # Negative Log-Likelihood
    brier_score: float  # Mean squared difference between prob and binary label
    n_bins: int
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_calibration_metrics(
    probabilities: Union[np.ndarray, List[float], torch.Tensor],
    labels: Union[np.ndarray, List[int], torch.Tensor],
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute Expected Calibration Error (ECE), MCE, and reliability diagram bins.

    Confidence is defined as the maximum predicted class probability:
    conf = max(p, 1 - p), and accuracy is whether the predicted class matches the true label.

    Args:
        probabilities: Predicted probabilities of class 1 in [0.0, 1.0].
        labels: Binary ground truth labels (0 or 1).
        n_bins: Number of equal-width confidence bins.

    Returns:
        CalibrationMetrics instance.
    """
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.detach().cpu().numpy().flatten()
    else:
        probs = np.asarray(probabilities, dtype=np.float64).flatten()

    if isinstance(labels, torch.Tensor):
        labs = labels.detach().cpu().numpy().flatten()
    else:
        labs = np.asarray(labels, dtype=np.int64).flatten()

    if len(probs) != len(labs):
        raise ValueError(f"Length mismatch: {len(probs)} probabilities vs {len(labs)} labels")

    if len(probs) == 0:
        return CalibrationMetrics(
            ece=0.0, mce=0.0, nll=0.0, brier_score=0.0, n_bins=n_bins,
            bin_accuracies=[], bin_confidences=[], bin_counts=[]
        )

    # Class predictions and confidence: conf = max(p, 1 - p)
    preds = (probs >= 0.5).astype(np.int64)
    confidences = np.maximum(probs, 1.0 - probs)
    accuracies = (preds == labs).astype(np.float64)

    # Negative log-likelihood and Brier score
    clipped_probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    nll = float(log_loss(labs, clipped_probs, labels=[0, 1]))
    brier = float(np.mean((probs - labs) ** 2))

    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    bin_accs: List[float] = []
    bin_confs: List[float] = []
    bin_counts: List[int] = []

    ece = 0.0
    mce = 0.0
    total_samples = len(probs)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        count = int(np.sum(in_bin))
        bin_counts.append(count)

        if count > 0:
            avg_acc = float(np.mean(accuracies[in_bin]))
            avg_conf = float(np.mean(confidences[in_bin]))
            bin_accs.append(round(avg_acc, 4))
            bin_confs.append(round(avg_conf, 4))

            abs_diff = abs(avg_acc - avg_conf)
            ece += (count / total_samples) * abs_diff
            mce = max(mce, abs_diff)
        else:
            bin_accs.append(0.0)
            bin_confs.append(round(float((bin_lower + bin_upper) / 2.0), 4))

    return CalibrationMetrics(
        ece=round(float(ece), 4),
        mce=round(float(mce), 4),
        nll=round(nll, 4),
        brier_score=round(brier, 4),
        n_bins=n_bins,
        bin_accuracies=bin_accs,
        bin_confidences=bin_confs,
        bin_counts=bin_counts,
    )


class TemperatureScaler(nn.Module):
    """Post-hoc probability calibration via temperature scaling (Guo et al., 2017).

    Operates strictly as a post-hoc scaling factor on logits:
    p_calibrated = sigmoid(z / T).
    Temperature T > 0 is learned by minimizing NLL strictly on validation data.
    """

    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temperature], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        temp = self.temperature.clamp(min=1e-3)
        return logits / temp

    def fit(
        self,
        val_logits: Union[np.ndarray, List[float], torch.Tensor],
        val_labels: Union[np.ndarray, List[int], torch.Tensor],
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Fit optimal temperature on validation logits and labels.

        Args:
            val_logits: Uncalibrated model logits from validation split.
            val_labels: True validation binary labels.
            lr: Learning rate for L-BFGS optimizer.
            max_iter: Maximum optimization steps.

        Returns:
            Learned scalar temperature T.
        """
        if isinstance(val_logits, torch.Tensor):
            logits_t = val_logits.clone().detach().view(-1, 1).float()
        else:
            logits_t = torch.tensor(val_logits, dtype=torch.float32).view(-1, 1)

        if isinstance(val_labels, torch.Tensor):
            labels_t = val_labels.clone().detach().view(-1, 1).float()
        else:
            labels_t = torch.tensor(val_labels, dtype=torch.float32).view(-1, 1)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits_t)
            loss = criterion(scaled_logits, labels_t)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())

    def calibrate_probabilities(
        self,
        logits: Union[np.ndarray, List[float], torch.Tensor],
    ) -> np.ndarray:
        """Compute calibrated probabilities from logits using fitted temperature."""
        if isinstance(logits, torch.Tensor):
            logits_t = logits.clone().detach().view(-1, 1).float()
        else:
            logits_t = torch.tensor(logits, dtype=torch.float32).view(-1, 1)

        with torch.no_grad():
            scaled = self.forward(logits_t)
            cal_probs = torch.sigmoid(scaled).numpy().flatten()
        return cal_probs


def find_optimal_thresholds(
    probabilities: Union[np.ndarray, List[float]],
    labels: Union[np.ndarray, List[int]],
    grid_steps: int = 200,
) -> Dict[str, float]:
    """Search for optimal decision thresholds strictly on validation data.

    Evaluates:
    1. Youden's J statistic: max(Sensitivity + Specificity - 1) = max(TPR - FPR).
    2. F1-maximizing threshold: argmax F1(theta).

    Args:
        probabilities: Predicted probabilities of class 1.
        labels: Ground truth binary labels.
        grid_steps: Number of threshold evaluation points.

    Returns:
        Dictionary with optimal thresholds and validation performance at those thresholds.
    """
    probs = np.asarray(probabilities, dtype=np.float64).flatten()
    labs = np.asarray(labels, dtype=np.int64).flatten()

    thresholds = np.linspace(0.01, 0.99, grid_steps)
    best_youden_j = -1.0
    best_youden_thresh = 0.5
    best_f1 = -1.0
    best_f1_thresh = 0.5

    n_pos = np.sum(labs == 1)
    n_neg = np.sum(labs == 0)

    if n_pos == 0 or n_neg == 0:
        return {
            "youden_threshold": 0.5,
            "youden_j": 0.0,
            "f1_threshold": 0.5,
            "f1_score": 0.0,
        }

    for t in thresholds:
        preds = (probs >= t).astype(np.int64)
        tp = np.sum((preds == 1) & (labs == 1))
        fp = np.sum((preds == 1) & (labs == 0))
        fn = np.sum((preds == 0) & (labs == 1))
        tn = np.sum((preds == 0) & (labs == 0))

        tpr = tp / n_pos
        fpr = fp / n_neg
        youden_j = tpr - fpr

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if youden_j > best_youden_j:
            best_youden_j = youden_j
            best_youden_thresh = float(t)

        if f1 > best_f1:
            best_f1 = f1
            best_f1_thresh = float(t)

    return {
        "youden_threshold": round(best_youden_thresh, 4),
        "youden_j": round(float(best_youden_j), 4),
        "f1_threshold": round(best_f1_thresh, 4),
        "f1_score": round(float(best_f1), 4),
    }
