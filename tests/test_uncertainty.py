"""Unit tests for calibrated uncertainty, OOD feature distance, and three-state rejection policy."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.inference.uncertainty import (
    CalibratedDecisionPolicy,
    MCDropoutEstimator,
    OODFeatureEstimator,
    QualityAssessor,
    QualitySignals,
    UncertaintySignals,
)
from src.models.detector import DeepfakeDetector
from src.utils.config import ModelConfig


class DummyDetector(nn.Module):
    """Minimal detector stub with dropout for fast unit testing."""

    def __init__(self, feature_dim: int = 1280):
        super().__init__()
        self.feature_dim = feature_dim
        self.backbone_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor, return_embeddings: bool = False):
        b, t, c, h, w = x.shape
        # Create dummy temporal features
        feat = torch.ones(b, self.feature_dim, device=x.device)
        feat_drop = self.dropout(feat)
        logits = self.classifier(feat_drop)
        if return_embeddings:
            return logits, feat
        return logits


def test_mc_dropout_estimator():
    model = DummyDetector(feature_dim=64)
    estimator = MCDropoutEstimator(model=model, num_samples=10)

    x = torch.randn(1, 16, 3, 112, 112)
    mean_p, var_p, entropy, feat = estimator.estimate(x, device=torch.device("cpu"))

    assert 0.0 <= mean_p <= 1.0
    assert var_p >= 0.0
    assert entropy >= 0.0
    assert feat.shape == (64,)


def test_ood_feature_estimator(tmp_path):
    estimator = OODFeatureEstimator(feature_dim=32)

    # Generate synthetic in-distribution data ~ N(0, 1)
    np.random.seed(42)
    in_dist_data = np.random.randn(100, 32).astype(np.float32)

    estimator.fit(in_dist_data, std_threshold=3.0)
    assert estimator.is_fitted

    # An in-distribution sample should have a small distance
    test_in = np.random.randn(32).astype(np.float32)
    dist_in = estimator.compute_distance(test_in)
    assert dist_in < 3.0
    assert not estimator.is_ood(dist_in)

    # An anomalous OOD sample with huge shift
    test_ood = np.full(32, 10.0, dtype=np.float32)
    dist_ood = estimator.compute_distance(test_ood)
    assert dist_ood > 3.0
    assert estimator.is_ood(dist_ood)

    # Save and load verification
    save_file = tmp_path / "ood_ref.json"
    estimator.save(save_file)

    loaded_est = OODFeatureEstimator(feature_dim=32)
    loaded_est.load(save_file)
    assert loaded_est.is_fitted
    assert loaded_est.compute_distance(test_ood) == pytest.approx(dist_ood, rel=1e-4)


def test_quality_assessor():
    assessor = QualityAssessor(max_fallback_ratio=0.50, min_detected_frames=8)

    # 1. High quality
    good_signals = assessor.assess(
        total_frames=16,
        detected_frames=15,
        previous_bbox_frames=1,
        center_crop_frames=0,
        failed_frames=0,
    )
    assert good_signals.is_acceptable
    assert good_signals.rejection_reason is None
    assert good_signals.fallback_ratio == pytest.approx(1 / 16, abs=1e-3)

    # 2. Excessive fallback
    bad_signals_fallback = assessor.assess(
        total_frames=16,
        detected_frames=6,
        previous_bbox_frames=10,
        center_crop_frames=0,
        failed_frames=0,
    )
    assert not bad_signals_fallback.is_acceptable
    assert "Excessive fallback" in bad_signals_fallback.rejection_reason

    # 3. Corrupted / failed frames
    bad_signals_corrupt = assessor.assess(
        total_frames=16,
        detected_frames=14,
        previous_bbox_frames=0,
        center_crop_frames=0,
        failed_frames=2,
    )
    assert not bad_signals_corrupt.is_acceptable
    assert "corrupted/empty" in bad_signals_corrupt.rejection_reason


def test_calibrated_decision_policy():
    policy = CalibratedDecisionPolicy(
        classification_threshold=0.50,
        max_uncertainty_variance=0.04,
        ood_distance_threshold=3.5,
    )

    clean_quality = QualitySignals(
        total_frames=16,
        detected_frames=16,
        fallback_frames=0,
        failed_frames=0,
        fallback_ratio=0.0,
        is_acceptable=True,
    )
    unacceptable_quality = QualitySignals(
        total_frames=16,
        detected_frames=4,
        fallback_frames=12,
        failed_frames=0,
        fallback_ratio=0.75,
        is_acceptable=False,
        rejection_reason="Excessive fallback crops",
    )

    clean_uncertainty = UncertaintySignals(
        mean_probability=0.85,
        epistemic_variance=0.01,
        entropy=0.4,
        mc_samples=10,
        ood_distance=1.2,
        is_ood=False,
        is_uncertain=False,
    )

    ood_uncertainty = UncertaintySignals(
        mean_probability=0.85,
        epistemic_variance=0.01,
        entropy=0.4,
        mc_samples=10,
        ood_distance=4.8,
        is_ood=True,
        is_uncertain=False,
    )

    high_variance_uncertainty = UncertaintySignals(
        mean_probability=0.52,
        epistemic_variance=0.08,  # > 0.04
        entropy=0.99,
        mc_samples=10,
        ood_distance=1.5,
        is_ood=False,
        is_uncertain=True,
    )

    # Case 1: Clean high-confidence fake
    v, conf, code, reason = policy.decide(0.85, clean_quality, clean_uncertainty)
    assert v == "Deepfake"
    assert conf == 0.85
    assert code is None
    assert reason is None

    # Case 2: Clean high-confidence real
    v, conf, code, reason = policy.decide(0.15, clean_quality, clean_uncertainty)
    assert v == "Real"
    assert conf == 0.85
    assert code is None
    assert reason is None

    # Case 3: Rejected due to quality
    v, conf, code, reason = policy.decide(0.90, unacceptable_quality, clean_uncertainty)
    assert v == "Reject"
    assert code == "LOW_QUALITY"
    assert "Low Video Quality" in reason

    # Case 4: Rejected due to OOD
    v, conf, code, reason = policy.decide(0.90, clean_quality, ood_uncertainty)
    assert v == "Reject"
    assert code == "OOD_HIGH"
    assert "Out-of-Distribution" in reason

    # Case 5: Rejected due to high epistemic uncertainty
    v, conf, code, reason = policy.decide(0.52, clean_quality, high_variance_uncertainty)
    assert v == "Reject"
    assert code == "HIGH_UNCERTAINTY"
    assert "High Epistemic Uncertainty" in reason

    # Case 6: Compound rejection (both low quality and high uncertainty)
    v, conf, code, reason = policy.decide(0.52, unacceptable_quality, high_variance_uncertainty)
    assert v == "Reject"
    assert code == "LOW_QUALITY_AND_HIGH_UNCERTAINTY"
    assert "Low Video Quality" in reason
    assert "High Epistemic Uncertainty" in reason
