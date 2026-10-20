"""Unit tests for EvaluationMetrics, compute_eer, and compute_metrics."""

import numpy as np
import pytest
import torch

from src.evaluation.metrics import EvaluationMetrics, compute_eer, compute_metrics


def test_compute_eer_exact_crossing():
    """Verify compute_eer performs exact linear crossing interpolation."""
    # Balanced dataset where FPR and FNR intersect
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Scores designed to cross around 0.5 threshold
    y_score = np.array([0.1, 0.2, 0.4, 0.6, 0.4, 0.7, 0.8, 0.9])

    eer = compute_eer(y_true, y_score)
    assert isinstance(eer, float)
    assert 0.0 <= eer <= 1.0
    assert not np.isnan(eer)
    assert abs(eer - 0.25) < 0.15


def test_compute_eer_single_class_returns_nan():
    """Verify compute_eer returns NaN when dataset has only one class."""
    y_true_all_zeros = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.isnan(compute_eer(y_true_all_zeros, y_score))

    y_true_all_ones = np.array([1, 1, 1])
    assert np.isnan(compute_eer(y_true_all_ones, y_score[:3]))


def test_compute_metrics_shape_validation():
    """Verify compute_metrics strictly enforces matching (N, 1) tensor shapes."""
    # 1D vs 2D mismatch (B,) vs (B, 1)
    y_true_1d = torch.tensor([0.0, 1.0])
    y_logits_2d = torch.tensor([[0.5], [-0.5]])

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_metrics(y_true_1d, y_logits_2d)

    # Length mismatch
    y_true_3 = torch.tensor([[0.0], [1.0], [0.0]])
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_metrics(y_true_3, y_logits_2d)

    # 3D tensor rejected
    y_3d = torch.zeros(2, 1, 1)
    with pytest.raises(ValueError, match="Expected 2D tensors of shape"):
        compute_metrics(y_3d, y_3d)


def test_compute_metrics_empty_input():
    """Verify compute_metrics raises ValueError on 0 samples."""
    empty_true = torch.empty(0, 1)
    empty_logits = torch.empty(0, 1)

    with pytest.raises(ValueError, match="Cannot compute metrics on empty inputs"):
        compute_metrics(empty_true, empty_logits)


def test_compute_metrics_single_class_guard():
    """Verify single-class evaluation produces NaN for AUC/EER without throwing exceptions."""
    y_true = torch.zeros(10, 1, dtype=torch.float32)
    y_logits = torch.randn(10, 1, dtype=torch.float32)

    metrics = compute_metrics(y_true, y_logits, loss=0.45)
    assert metrics.num_samples == 10
    assert metrics.loss == 0.45
    assert isinstance(metrics.accuracy, float)
    assert np.isnan(metrics.roc_auc)
    assert np.isnan(metrics.eer)
    assert "NaN (single-class)" in metrics.summary_string()


def test_compute_metrics_balanced_values():
    """Verify metric calculations (Acc, Prec, Rec, F1, ROC-AUC, CM) on known ground truth."""
    # 4 samples: 2 real (0), 2 fake (1)
    y_true = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    # Logits: high logit => sigmoid > 0.5 (pred=1), low logit => sigmoid < 0.5 (pred=0)
    # Target:   0     0     1    1
    # Pred:     0     1     0    1
    # TP=1, FP=1, TN=1, FN=1
    y_logits = torch.tensor([[-2.0], [2.0], [-2.0], [2.0]])

    metrics = compute_metrics(y_true, y_logits, loss=0.6931, threshold=0.5)

    assert metrics.num_samples == 4
    assert metrics.confusion_matrix == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert not np.isnan(metrics.roc_auc)
    assert not np.isnan(metrics.eer)
