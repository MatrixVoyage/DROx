"""Unit tests for Milestone 9 statistical significance, video-level bootstrap & paired tests."""

import numpy as np
import pytest

from src.evaluation.statistics import (
    bootstrap_video_level_metrics,
    compute_paired_mcnemar_test,
)


def test_video_level_bootstrap_ci():
    # 20 samples
    probs = [0.8, 0.9, 0.7, 0.6, 0.85, 0.95, 0.75, 0.65, 0.88, 0.92,
             0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.12, 0.22, 0.32, 0.18]
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    res = bootstrap_video_level_metrics(
        probabilities=probs,
        labels=labels,
        n_bootstraps=100,
        confidence_level=0.95,
        random_seed=42,
    )

    assert res["resampling_unit"] == "video"
    assert res["random_seed"] == 42
    assert "metrics" in res
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        m_ci = res["metrics"][m]
        assert m_ci["ci_lower"] <= m_ci["point_estimate"] + 1e-4
        assert m_ci["ci_upper"] >= m_ci["point_estimate"] - 1e-4


def test_paired_mcnemar_test():
    ground_truth = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    base_preds   = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]  # 6 correct
    alt_preds    = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 10 correct

    res = compute_paired_mcnemar_test(
        baseline_predictions=base_preds,
        alternative_predictions=alt_preds,
        ground_truth=ground_truth,
        baseline_name="baseline_model",
        alternative_name="improved_model",
    )

    assert "contingency_table" in res
    assert res["contingency_table"]["both_correct"] == 6
    assert res["contingency_table"]["alternative_only_correct"] == 4
    assert res["contingency_table"]["baseline_only_correct"] == 0
    assert "p_value" in res
    assert res["p_value"] <= 1.0
