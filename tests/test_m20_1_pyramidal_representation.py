"""Automated unit tests for Milestone 20.1: Whole-Face Multi-Scale Pyramidal Diagnostic.

Verifies:
1. Laplacian pyramid mathematical decomposition and exact reconstruction invariants.
2. Arm tensor generation and shape invariants (1, 16, 3, 112, 112).
3. Numerical bounds: finite, no NaNs, valid normalization.
4. Paired bootstrap statistics and outcome classification logic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.diagnose_m20_1_pyramidal_representation import (
    decompose_laplacian_pyramid,
    construct_pyramidal_arm_patches,
    normalize_pyramidal_sequence,
    compute_paired_bootstrap,
    compute_cliffs_delta,
    classify_pyramidal_outcome,
)


def test_laplacian_pyramid_reconstruction_invariant():
    """Verify that Gaussian downsampling and Laplacian detail layers reconstruct perfectly."""
    # Synthetic native face crop: 224 x 224 x 3 uint8
    H0, W0 = 224, 224
    rng = np.random.RandomState(42)
    G0 = rng.randint(0, 256, (H0, W0, 3), dtype=np.uint8)

    pyramid = decompose_laplacian_pyramid(G0, target_size=112)

    assert "G0" in pyramid
    assert "G1" in pyramid
    assert "G2" in pyramid
    assert "L0" in pyramid
    assert "L1" in pyramid

    assert pyramid["G1"].shape == (112, 112, 3)
    assert pyramid["G2"].shape == (56, 56, 3)
    assert pyramid["L0"].shape == (H0, W0, 3)
    assert pyramid["L1"].shape == (112, 112, 3)

    # Check G1 reconstruction: G2_up + L1 == G1 (in float space)
    # Using float reconstruction
    assert "G1_recon_err" in pyramid
    assert pyramid["G1_recon_err"] < 1e-4, f"G1 reconstruction error too high: {pyramid['G1_recon_err']}"


def test_pyramidal_arms_construction_and_shapes():
    """Verify that all diagnostic arms produce valid 112x112 frames."""
    H0, W0 = 200, 200
    G0 = np.random.randint(0, 256, (H0, W0, 3), dtype=np.uint8)
    pyramid = decompose_laplacian_pyramid(G0, target_size=112)

    arms = construct_pyramidal_arm_patches(pyramid)
    expected_arms = ["arm_1_base", "arm_2_mid_l1", "arm_3_high_l0", "arm_4_inject_02", "arm_4_inject_05"]

    for arm_name in expected_arms:
        assert arm_name in arms, f"Missing arm: {arm_name}"
        assert arms[arm_name].shape == (112, 112, 3), f"Wrong shape for {arm_name}: {arms[arm_name].shape}"
        assert arms[arm_name].dtype == np.uint8, f"Wrong dtype for {arm_name}: {arms[arm_name].dtype}"


def test_pyramidal_sequence_normalization_invariants():
    """Verify that sequence tensors normalize to (1, 16, 3, 112, 112) float32 with no NaNs/Infs."""
    raw_patches = [np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8) for _ in range(16)]
    tensor = normalize_pyramidal_sequence(raw_patches)

    assert tensor.shape == (1, 16, 3, 112, 112)
    assert tensor.dtype == torch.float32
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()


def test_paired_bootstrap_statistics():
    """Verify paired bootstrap calculation on synthetic predictions."""
    np.random.seed(42)
    y_true = np.array([0] * 50 + [1] * 50)
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


def test_pyramidal_outcome_classification_logic():
    """Verify classification across Support, Inconclusive, and Negative Evidence."""
    # Case 1: Support (Arm 4 injection substantially improves over Arm 1)
    metrics_support = {
        "arm_1_base": {"auc": 0.7669, "logit_margin": 0.380},
        "arm_2_mid_l1": {"auc": 0.6200, "logit_margin": 0.150},
        "arm_3_high_l0": {"auc": 0.6500, "logit_margin": 0.180},
        "arm_4_inject_02": {"auc": 0.8050, "delta_auc_vs_base": +0.0381, "p_val_vs_base": 0.020, "logit_margin": 0.450},
        "arm_4_inject_05": {"auc": 0.7900, "delta_auc_vs_base": +0.0231, "p_val_vs_base": 0.040, "logit_margin": 0.410},
    }
    res_1 = classify_pyramidal_outcome(metrics_support)
    assert res_1["tier"] == "SUPPORT"

    # Case 2: Inconclusive / Frequency-Shift (Isolated Laplacian drops, but Injection holds margin)
    metrics_inconclusive = {
        "arm_1_base": {"auc": 0.7669, "logit_margin": 0.380},
        "arm_2_mid_l1": {"auc": 0.5200, "logit_margin": -0.050},
        "arm_3_high_l0": {"auc": 0.5100, "logit_margin": -0.080},
        "arm_4_inject_02": {"auc": 0.7650, "delta_auc_vs_base": -0.0019, "p_val_vs_base": 0.520, "logit_margin": 0.370},
        "arm_4_inject_05": {"auc": 0.7500, "delta_auc_vs_base": -0.0169, "p_val_vs_base": 0.750, "logit_margin": 0.340},
    }
    res_2 = classify_pyramidal_outcome(metrics_inconclusive)
    assert res_2["tier"] in ["INCONCLUSIVE", "NEGATIVE EVIDENCE"]
