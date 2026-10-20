"""Unit tests for Milestone 21: Locked Face2Face Failure Analysis & Evidence Attribution.

Verifies:
1. ROI-restricted optical flow computation and non-negative scalar output.
2. DCT high-frequency energy ratio bounds in [0, 1].
3. Geometric facial scale extraction (inter-eye distance, frame fraction).
4. Boundary seam Laplacian energy and color Delta E.
5. Semantic mouth activity metrics (amplitude, velocity, cycles).
6. Multi-model cohort partitioning and 4-bit discordance matrix vectoring.
7. Minimum-N gating (N >= 8 for inferential tests, descriptive fallback).
8. Within-family Benjamini-Hochberg FDR correction.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_m21_f2f_failures import (
    compute_roi_optical_flow,
    compute_dct_high_frequency_ratio,
    compute_facial_scale_metrics,
    compute_seam_gradient_metrics,
    compute_semantic_activity_metrics,
    partition_prediction_cohorts,
    evaluate_minimum_n_gating,
    apply_family_fdr,
)


def test_compute_roi_optical_flow():
    """Verify Farneback optical flow restricted to lower-face ROI returns valid magnitude."""
    # Create two synthetic frames with horizontal motion
    f1 = np.zeros((100, 100, 3), dtype=np.uint8)
    f2 = np.zeros((100, 100, 3), dtype=np.uint8)
    f1[30:70, 30:70] = 255
    f2[30:70, 35:75] = 255  # 5 pixel horizontal shift

    # Lower face ROI: rows 50 to 90, cols 20 to 80
    roi = (20, 50, 80, 90)  # x1, y1, x2, y2
    flow_mag = compute_roi_optical_flow(f1, f2, roi)

    assert isinstance(flow_mag, float)
    assert flow_mag > 0.0
    assert not np.isnan(flow_mag)

    # Static frames should yield zero/near-zero flow
    flow_static = compute_roi_optical_flow(f1, f1, roi)
    assert flow_static < 1e-4


def test_compute_dct_high_frequency_ratio():
    """Verify 2D DCT high-frequency energy ratio satisfies bounds [0, 1]."""
    # Smooth patch (mostly low frequency)
    smooth = np.ones((64, 64), dtype=np.float32) * 128.0
    ratio_smooth = compute_dct_high_frequency_ratio(smooth)
    assert 0.0 <= ratio_smooth <= 1.0
    assert ratio_smooth < 0.1

    # High frequency checkerboard patch
    checker = np.indices((64, 64)).sum(axis=0) % 2 * 255.0
    ratio_high = compute_dct_high_frequency_ratio(checker.astype(np.float32))
    assert 0.0 <= ratio_high <= 1.0
    assert ratio_high > ratio_smooth


def test_compute_facial_scale_metrics():
    """Verify scale metrics and inter-eye distance extraction."""
    # Mock bbox (x1, y1, x2, y2) and landmarks (left_eye, right_eye)
    bbox = (100, 100, 300, 350)  # width 200, height 250
    landmarks = {
        "left_eye": (150.0, 160.0),
        "right_eye": (250.0, 160.0),
    }
    frame_size = (1920, 1080)  # W, H

    metrics = compute_facial_scale_metrics(bbox, landmarks, frame_size)
    assert metrics["face_width"] == 200
    assert metrics["face_height"] == 250
    assert metrics["face_area"] == 50000
    assert abs(metrics["inter_eye_distance"] - 100.0) < 1e-4
    assert 0.0 < metrics["face_fraction_of_frame"] < 1.0


def test_compute_seam_gradient_metrics():
    """Verify boundary seam gradient and color Delta E metrics."""
    # Synthetic frame with a distinct circular composite seam
    patch = np.zeros((112, 112, 3), dtype=np.uint8)
    # Inner face white, outer background black
    patch[25:85, 25:85] = 200

    metrics = compute_seam_gradient_metrics(patch)
    assert "seam_laplacian_energy" in metrics
    assert "seam_color_delta_e" in metrics
    assert metrics["seam_laplacian_energy"] >= 0.0
    assert metrics["seam_color_delta_e"] >= 0.0


def test_compute_semantic_activity_metrics():
    """Verify measurable mouth and eye activity metrics."""
    # Sequence of 16 mouth heights simulating speech opening/closing
    mouth_heights = [10, 12, 18, 25, 22, 15, 10, 8, 12, 20, 24, 18, 12, 10, 9, 10]
    inter_eye_dist = 100.0
    eye_aspect_ratios = [0.30] * 16
    eye_aspect_ratios[8] = 0.05  # single blink

    activity = compute_semantic_activity_metrics(mouth_heights, eye_aspect_ratios, inter_eye_dist)
    assert activity["mouth_opening_amplitude"] == pytest.approx((25.0 - 8.0) / 100.0)
    assert activity["mouth_opening_velocity"] > 0.0
    assert activity["mouth_motion_cycles"] >= 1
    assert activity["eye_opening_variance"] > 0.0


def test_partition_prediction_cohorts():
    """Verify multi-model cohort partitioning and 4-bit discordance matrix."""
    items = [
        # Clip 1: Detected Fake (all 4 say Fake)
        {"id": "c1", "label": 1, "preds": {"m13_5": 1, "m17": 1, "m18": 1, "m20_2": 1}, "logits": {}},
        # Clip 2: Hard FN (all 4 say Real)
        {"id": "c2", "label": 1, "preds": {"m13_5": 0, "m17": 0, "m18": 0, "m20_2": 0}, "logits": {}},
        # Clip 3: M13.5 Win Fake
        {"id": "c3", "label": 1, "preds": {"m13_5": 1, "m17": 0, "m18": 0, "m20_2": 0}, "logits": {}},
        # Clip 4: Clean Real (all 4 say Real)
        {"id": "c4", "label": 0, "preds": {"m13_5": 0, "m17": 0, "m18": 0, "m20_2": 0}, "logits": {}},
        # Clip 5: Hard FP (all 4 say Fake)
        {"id": "c5", "label": 0, "preds": {"m13_5": 1, "m17": 1, "m18": 1, "m20_2": 1}, "logits": {}},
    ]

    cohorts, discordance = partition_prediction_cohorts(items)
    assert "c1" in cohorts["cohort_1_detected_fakes"]
    assert "c2" in cohorts["cohort_2_hard_fns"]
    assert "c4" in cohorts["cohort_3_clean_reals"]
    assert "c5" in cohorts["cohort_4_hard_fps"]
    assert discordance["c3"] == [1, 0, 0, 0]


def test_evaluate_minimum_n_gating():
    """Verify minimum-N rule gates inferential testing at N < 8."""
    # N = 5 should fail minimum N check
    assert not evaluate_minimum_n_gating(n_samples=5, min_n=8)
    # N = 8 should pass minimum N check
    assert evaluate_minimum_n_gating(n_samples=8, min_n=8)
    assert evaluate_minimum_n_gating(n_samples=15, min_n=8)


def test_apply_family_fdr():
    """Verify Benjamini-Hochberg FDR control within a single hypothesis family."""
    p_values = [0.001, 0.02, 0.04, 0.30, 0.80]
    q_values = apply_family_fdr(p_values)
    assert len(q_values) == len(p_values)
    assert q_values[0] <= q_values[1] <= q_values[2]
    assert all(0.0 <= q <= 1.0 for q in q_values)
