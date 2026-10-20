"""Unit tests for Milestone 20.2 Calibration and 7-Gate Promotion Evaluation logic.

Verifies:
1. PlattScaler, TemperatureScaler, and IsotonicCalibrator mathematical invariants.
2. Raw logit profiling and distribution margin metrics.
3. Validation-derived threshold policy derivation (tau_95*, tau_99*, Youden, F1).
4. ECE, Adaptive ECE, and MCE error metric bounds.
5. Paired bootstrap statistics and confidence interval ordering.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.calibrate_m20_decision_layer import (
    PlattScaler,
    TemperatureScaler,
    IsotonicCalibrator,
    derive_threshold_policies,
    profile_raw_logits,
    compute_ece,
    compute_adaptive_ece,
    compute_mce,
)
from scripts.evaluate_m20_promotions import compute_paired_bootstrap


def test_m20_scalers_invariants():
    """Verify Temperature and Platt scalers fit and transform monotonically."""
    np.random.seed(42)
    logits = np.random.randn(200) + 1.0
    labels = (logits > 0.5).astype(int)

    # Temperature Scaler
    t_scaler = TemperatureScaler().fit(logits, labels)
    assert t_scaler.is_fitted
    assert 0.1 <= t_scaler.T <= 10.0
    probs_t = t_scaler.transform(logits)
    assert (probs_t >= 0.0).all() and (probs_t <= 1.0).all()
    # Monotonicity
    assert np.all(np.diff(probs_t[np.argsort(logits)]) >= 0.0)

    # Platt Scaler
    p_scaler = PlattScaler().fit(logits, labels)
    assert p_scaler.is_fitted
    probs_p = p_scaler.transform(logits)
    assert (probs_p >= 0.0).all() and (probs_p <= 1.0).all()
    # Monotonicity
    assert np.all(np.diff(probs_p[np.argsort(logits)]) >= 0.0)

    # Isotonic Calibrator
    raw_probs = 1.0 / (1.0 + np.exp(-logits))
    iso_cal = IsotonicCalibrator().fit(raw_probs, labels)
    assert iso_cal.is_fitted
    probs_iso = iso_cal.transform(raw_probs)
    assert (probs_iso >= 0.0).all() and (probs_iso <= 1.0).all()


def test_m20_logit_profiling():
    """Verify raw logit distribution profiling."""
    reals = np.array([-2.0, -1.0, -0.5, 0.0])
    fakes = np.array([0.5, 1.0, 2.0, 3.0])
    logits = np.concatenate([reals, fakes])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    prof = profile_raw_logits(logits, labels)
    assert prof["real_median"] < prof["fake_median"]
    assert prof["logit_margin"] > 0.0
    assert 0.0 <= prof["distribution_overlap_pct"] <= 1.0


def test_m20_threshold_policy_derivation():
    """Verify threshold policies satisfy target constraints."""
    np.random.seed(42)
    p_clean = np.random.uniform(0.01, 0.40, size=100)
    p_fake = np.random.uniform(0.60, 0.99, size=100)
    probs = np.concatenate([p_clean, p_fake])
    labels = np.concatenate([np.zeros(100), np.ones(100)])

    policies = derive_threshold_policies(probs, labels)
    assert "tau_95" in policies
    assert "tau_99" in policies
    assert "tau_youden" in policies
    assert "tau_f1" in policies

    # Specificity check under tau_95
    spec_95 = np.mean(p_clean < policies["tau_95"])
    assert spec_95 >= 0.95


def test_m20_calibration_metrics_bounds():
    """Verify ECE and MCE metrics adhere to [0, 1]."""
    probs = np.linspace(0.1, 0.9, 100)
    labels = (probs > 0.5).astype(int)

    ece = compute_ece(probs, labels, n_bins=10)
    a_ece = compute_adaptive_ece(probs, labels, n_bins=10)
    mce = compute_mce(probs, labels, n_bins=10)

    assert 0.0 <= ece <= 1.0
    assert 0.0 <= a_ece <= 1.0
    assert 0.0 <= mce <= 1.0


def test_m20_paired_bootstrap():
    """Verify paired bootstrap returns valid confidence intervals and p-value."""
    y_true = np.array([0]*50 + [1]*50)
    scores_cand = np.array([0.1]*50 + [0.9]*50)
    scores_ref = np.array([0.4]*50 + [0.6]*50)

    bs = compute_paired_bootstrap(y_true, scores_cand, scores_ref, n_resamples=200, seed=42)
    assert bs["ci_lower"] <= bs["delta_mean"] <= bs["ci_upper"]
    assert 0.0 <= bs["p_value"] <= 1.0
    assert bs["delta_mean"] >= 0.0
