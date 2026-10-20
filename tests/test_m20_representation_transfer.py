"""Unit tests for Milestone 20 Phase 1: Frozen-Weight Localized Representation Transfer Ablation.

Verifies:
1. Extraction geometry and scale matching: Arm E1 exactly matches Arm B size, Arm E2 matches Arm C size.
2. Tensor shape and normalization invariants: (16, 3, 112, 112), float32, no NaNs.
3. Paired bootstrap statistics and non-parametric effect sizes.
4. Three-tier outcome classification logic (Support, Inconclusive, Negative Evidence).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.ablate_m20_representation_transfer import (
    extract_anatomical_patches,
    compute_paired_bootstrap,
    compute_cliffs_delta,
    classify_phase1_outcome,
)


def test_extraction_geometry_and_scale_matching():
    """Verify that negative controls E1 and E2 strictly match sizes of B and C."""
    # Synthetic frame: 720 x 1280
    frame_h, frame_w = 720, 1280
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    # Synthetic face bbox and landmarks
    bbox = np.array([500.0, 150.0, 750.0, 450.0]) # w=250, h=300
    # Landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
    landmarks = np.array([
        [570.0, 250.0], # left eye
        [680.0, 250.0], # right eye
        [625.0, 310.0], # nose
        [580.0, 370.0], # left mouth
        [670.0, 370.0], # right mouth
    ])

    patches, meta = extract_anatomical_patches(frame, bbox, landmarks)

    # Check all required keys exist
    expected_arms = ["arm_a", "arm_b", "arm_c", "arm_d", "arm_e1", "arm_e2"]
    for arm in expected_arms:
        assert arm in patches, f"Missing arm: {arm}"
        assert patches[arm].shape == (112, 112, 3), f"Incorrect shape for {arm}: {patches[arm].shape}"
        assert patches[arm].dtype == np.uint8

    # Scale matching invariants:
    # Size of E1 must equal size of B
    assert meta["size_b"] == meta["size_e1"], f"Scale mismatch: B={meta['size_b']} vs E1={meta['size_e1']}"
    # Size of E2 must equal size of C
    assert meta["size_c"] == meta["size_e2"], f"Scale mismatch: C={meta['size_c']} vs E2={meta['size_e2']}"
    assert meta["size_b"] > 0
    assert meta["size_c"] > 0


def test_tensor_normalization_invariants():
    """Verify extracted patch sequences normalize to (16, 3, 112, 112) without NaNs."""
    from scripts.ablate_m20_representation_transfer import normalize_patch_sequence

    # Generate synthetic 16 frames of uint8
    raw_patches = [np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8) for _ in range(16)]
    tensor = normalize_patch_sequence(raw_patches)

    assert tensor.shape == (1, 16, 3, 112, 112)
    assert tensor.dtype == torch.float32
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()


def test_paired_bootstrap_and_cliffs_delta():
    """Verify paired bootstrap statistical invariants and effect size calculation."""
    np.random.seed(42)
    y_true = np.array([0] * 50 + [1] * 50)
    # Cand has superior margin than Ref
    scores_ref = np.concatenate([np.random.normal(0.0, 1.0, 50), np.random.normal(1.0, 1.0, 50)])
    scores_cand = np.concatenate([np.random.normal(-0.5, 1.0, 50), np.random.normal(1.5, 1.0, 50)])

    res = compute_paired_bootstrap(y_true, scores_cand, scores_ref, n_resamples=500, seed=42)
    assert "delta_auc" in res
    assert "p_value" in res
    assert "ci_lower" in res
    assert "ci_upper" in res
    assert res["ci_lower"] <= res["delta_auc"] <= res["ci_upper"]

    d = compute_cliffs_delta(scores_cand[:50], scores_cand[50:])
    assert -1.0 <= d <= 1.0


def test_three_tier_outcome_classification():
    """Verify Three-Tier Outcome Logic (Support, Inconclusive, Negative Evidence)."""
    # Case 1: Support (Arm B substantially outperforms Arm A, and E1 does not)
    arm_metrics_support = {
        "arm_a": {"auc": 0.7854, "logit_margin": 0.547},
        "arm_b": {"auc": 0.8250, "delta_auc_vs_a": +0.0396, "p_val_vs_a": 0.015, "logit_margin": 0.750},
        "arm_c": {"auc": 0.7700, "delta_auc_vs_a": -0.0154, "p_val_vs_a": 0.700, "logit_margin": 0.400},
        "arm_d": {"auc": 0.7800, "delta_auc_vs_a": -0.0054, "p_val_vs_a": 0.550, "logit_margin": 0.510},
        "arm_e1": {"auc": 0.7400, "delta_auc_vs_a": -0.0454, "p_val_vs_a": 0.950, "logit_margin": 0.120},
        "arm_e2": {"auc": 0.7300, "delta_auc_vs_a": -0.0554, "p_val_vs_a": 0.980, "logit_margin": 0.100},
    }
    outcome_1 = classify_phase1_outcome(arm_metrics_support)
    assert outcome_1["tier"] == "SUPPORT"

    # Case 2: Negative Evidence (Control performs equivalently or better than anatomical arms)
    arm_metrics_negative = {
        "arm_a": {"auc": 0.7854, "logit_margin": 0.547},
        "arm_b": {"auc": 0.7200, "delta_auc_vs_a": -0.0654, "p_val_vs_a": 0.990, "logit_margin": 0.150},
        "arm_c": {"auc": 0.7100, "delta_auc_vs_a": -0.0754, "p_val_vs_a": 0.995, "logit_margin": 0.120},
        "arm_d": {"auc": 0.7150, "delta_auc_vs_a": -0.0704, "p_val_vs_a": 0.992, "logit_margin": 0.130},
        "arm_e1": {"auc": 0.7500, "delta_auc_vs_a": -0.0354, "p_val_vs_a": 0.850, "logit_margin": 0.250},
        "arm_e2": {"auc": 0.7400, "delta_auc_vs_a": -0.0454, "p_val_vs_a": 0.900, "logit_margin": 0.220},
    }
    outcome_2 = classify_phase1_outcome(arm_metrics_negative)
    assert outcome_2["tier"] in ["NEGATIVE EVIDENCE", "INCONCLUSIVE"]

    # Case 3: Inconclusive (Severe semantic distribution shift: logit margin completely collapses across all localized crops)
    arm_metrics_inconclusive = {
        "arm_a": {"auc": 0.7854, "logit_margin": 0.547},
        "arm_b": {"auc": 0.5100, "delta_auc_vs_a": -0.2754, "p_val_vs_a": 1.000, "logit_margin": -0.050},
        "arm_c": {"auc": 0.5050, "delta_auc_vs_a": -0.2804, "p_val_vs_a": 1.000, "logit_margin": -0.060},
        "arm_d": {"auc": 0.5120, "delta_auc_vs_a": -0.2734, "p_val_vs_a": 1.000, "logit_margin": -0.040},
        "arm_e1": {"auc": 0.5020, "delta_auc_vs_a": -0.2834, "p_val_vs_a": 1.000, "logit_margin": -0.070},
        "arm_e2": {"auc": 0.4980, "delta_auc_vs_a": -0.2874, "p_val_vs_a": 1.000, "logit_margin": -0.080},
    }
    outcome_3 = classify_phase1_outcome(arm_metrics_inconclusive)
    assert outcome_3["tier"] == "INCONCLUSIVE"
