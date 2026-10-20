"""Unit tests for systematic error analysis and detection correlation utilities."""

import pytest

from src.evaluation.error_analysis import (
    CorrelationFinding,
    ErrorSampleMetadata,
    analyze_detection_correlations,
    categorize_errors,
    compute_biserial_correlation,
    generate_error_analysis_summary,
)


def test_categorize_errors():
    """categorize_errors must accurately isolate False Positives and False Negatives."""
    predictions = [
        # True Real (label=0), predicted Real (prob=0.1) -> Correct
        {"video_id": "r1", "label": 0, "probability": 0.1, "dataset": "FF++", "manipulation": "original"},
        # True Real (label=0), predicted Fake (prob=0.8) -> FP
        {"video_id": "r2", "label": 0, "probability": 0.8, "dataset": "FF++", "manipulation": "original", "mtcnn_confidences": [0.99], "bboxes": [[10, 10, 50, 50]]},
        # True Fake (label=1), predicted Fake (prob=0.9) -> Correct
        {"video_id": "f1", "label": 1, "probability": 0.9, "dataset": "FF++", "manipulation": "Deepfakes"},
        # True Fake (label=1), predicted Real (prob=0.2) -> FN
        {"video_id": "f2", "label": 1, "probability": 0.2, "dataset": "FF++", "manipulation": "FaceSwap", "fallback_frames": 2, "num_frames": 16},
    ]

    fps, fns, correct = categorize_errors(predictions, threshold=0.5)

    assert len(fps) == 1
    assert fps[0].video_id == "r2"
    assert fps[0].error_type == "false_positive"
    assert fps[0].margin == 0.3
    assert fps[0].mean_bbox_area == 1600.0  # (50-10) * (50-10)

    assert len(fns) == 1
    assert fns[0].video_id == "f2"
    assert fns[0].error_type == "false_negative"
    assert fns[0].margin == 0.3
    assert fns[0].num_fallback_frames == 2

    assert len(correct) == 2


def test_compute_biserial_correlation():
    """Biserial correlation should handle constant inputs and perfect correlations."""
    # Constant input -> 0.0 correlation
    assert compute_biserial_correlation([5.0, 5.0, 5.0], [0, 1, 0]) == 0.0

    # Perfect correlation
    features = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
    errors = [0, 0, 0, 1, 1, 1]
    r = compute_biserial_correlation(features, errors)
    assert r > 0.85


def test_detection_correlations_non_causal_language():
    """Correlation findings must strictly employ non-causal terminology."""
    predictions = [
        {"video_id": f"v_{i}", "label": i % 2, "probability": 0.9 if i % 2 == 1 else 0.1, "mtcnn_confidences": [0.95], "bboxes": [[0, 0, 100, 100]]}
        for i in range(10)
    ]
    # Introduce one error
    predictions.append({"video_id": "err_1", "label": 0, "probability": 0.85, "mtcnn_confidences": [0.70], "bboxes": [[0, 0, 40, 40]]})

    findings = analyze_detection_correlations(predictions, threshold=0.5)
    assert len(findings) > 0

    for f in findings:
        assert isinstance(f, CorrelationFinding)
        # Check non-causal wording
        assert "causality" in f.interpretation or "associated with" in f.interpretation or "correlated" in f.interpretation
        assert "caused" not in f.interpretation.lower()


def test_generate_error_analysis_summary():
    """Summary must produce valid taxonomy and per-family statistics."""
    predictions = [
        {"video_id": "r1", "label": 0, "probability": 0.1, "manipulation": "original"},
        {"video_id": "r2", "label": 0, "probability": 0.9, "manipulation": "original"},
        {"video_id": "f1", "label": 1, "probability": 0.8, "manipulation": "Deepfakes"},
        {"video_id": "f2", "label": 1, "probability": 0.3, "manipulation": "Face2Face"},
    ]

    summary = generate_error_analysis_summary(predictions, threshold=0.5)
    assert summary["total_samples"] == 4
    assert summary["false_positive_count"] == 1
    assert summary["false_negative_count"] == 1
    assert summary["total_errors"] == 2
    assert "original" in summary["manipulation_family_errors"]
    assert "Face2Face" in summary["manipulation_family_errors"]
