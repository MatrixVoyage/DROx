"""Unit tests for probability calibration, temperature scaling, and threshold search."""

import numpy as np
import pytest
import torch

from src.evaluation.calibration import (
    CalibrationMetrics,
    TemperatureScaler,
    compute_calibration_metrics,
    find_optimal_thresholds,
)


def test_calibration_perfect_alignment():
    """Perfect predictions should yield near-zero calibration error."""
    probs = np.array([0.99, 0.95, 0.92, 0.05, 0.02, 0.01])
    labels = np.array([1, 1, 1, 0, 0, 0])

    metrics = compute_calibration_metrics(probs, labels, n_bins=5)
    assert isinstance(metrics, CalibrationMetrics)
    assert metrics.ece < 0.15
    assert metrics.brier_score < 0.05
    assert len(metrics.bin_counts) == 5


def test_calibration_miscalibrated():
    """Severely overconfident incorrect predictions should yield high ECE."""
    # High probability of class 1, but all true labels are 0
    probs = np.array([0.95, 0.92, 0.98, 0.89, 0.91])
    labels = np.array([0, 0, 0, 0, 0])

    metrics = compute_calibration_metrics(probs, labels, n_bins=5)
    assert metrics.ece > 0.70
    assert metrics.mce > 0.70
    assert metrics.brier_score > 0.80


def test_calibration_empty_and_mismatch():
    """Empty inputs and shape mismatches must be handled cleanly."""
    empty = compute_calibration_metrics([], [])
    assert empty.ece == 0.0
    assert empty.mce == 0.0

    with pytest.raises(ValueError, match="Length mismatch"):
        compute_calibration_metrics([0.5, 0.6], [1])


def test_temperature_scaler_optimization():
    """TemperatureScaler must optimize T strictly on validation logits via L-BFGS."""
    # Create overconfident logits
    torch.manual_seed(42)
    val_logits = torch.tensor([5.0, 4.0, 4.5, -4.0, -5.0, -3.5], dtype=torch.float32)
    val_labels = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float32)  # labels are noisy

    scaler = TemperatureScaler(init_temperature=1.0)
    assert scaler.temperature.item() == 1.0

    fitted_t = scaler.fit(val_logits, val_labels, max_iter=50)
    assert isinstance(fitted_t, float)
    assert fitted_t > 0.1  # Temperature must remain positive
    assert scaler.temperature.item() == fitted_t

    # Scaling transforms logits
    test_logits = torch.tensor([2.0, -2.0])
    scaled = scaler(test_logits)
    assert torch.is_tensor(scaled)
    assert scaled.shape == test_logits.shape


def test_find_optimal_thresholds():
    """Threshold search must identify valid Youden J and F1 maximizing thresholds."""
    probs = np.array([0.1, 0.2, 0.45, 0.55, 0.7, 0.85, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1, 1])

    threshold_res = find_optimal_thresholds(probs, labels)
    assert "youden_threshold" in threshold_res
    assert "youden_j" in threshold_res
    assert "f1_threshold" in threshold_res
    assert "f1_score" in threshold_res

    youden_thresh = threshold_res["youden_threshold"]
    f1_thresh = threshold_res["f1_threshold"]

    assert 0.0 <= youden_thresh <= 1.0
    assert 0.0 <= f1_thresh <= 1.0
    assert threshold_res["f1_score"] == 1.0
