"""Tests for EmpiricalBaselineManager, finite z-score computation, and quantile integrity."""
import json
from pathlib import Path
import pytest

from src.forensics.baseline import (
    compute_z_score,
    BaselineReference,
    EmpiricalBaselineManager,
)


def test_compute_z_score_finite_and_zero_variance():
    """Verify z-score returns None for zero variance and rejects non-finite values."""
    # Standard valid calculation
    z = compute_z_score(value=1.5, mean=1.0, std=0.25)
    assert pytest.approx(z, 0.001) == 2.0

    # Zero or negative variance: must return None (not estimable), not runaway huge numbers
    assert compute_z_score(value=1.5, mean=1.0, std=0.0) is None
    assert compute_z_score(value=1.5, mean=1.0, std=-0.5) is None

    # Non-finite values: must raise ValueError
    with pytest.raises(ValueError, match="must be finite"):
        compute_z_score(value=float("nan"), mean=1.0, std=0.2)
    with pytest.raises(ValueError, match="must be finite"):
        compute_z_score(value=1.0, mean=float("inf"), std=0.2)
    with pytest.raises(ValueError, match="must be finite"):
        compute_z_score(value=1.0, mean=0.0, std=float("inf"))


def test_baseline_reference_quantile_validation():
    """Verify BaselineReference enforces sample_count >= 100 and p05 <= median <= p95."""
    # Valid reference
    ref = BaselineReference(
        metric_name="motion_divergence",
        feature_version="1.0",
        dataset_name="FaceForensics++ C23",
        sample_video_count=120,
        mean=0.15,
        std=0.03,
        median=0.14,
        p05=0.10,
        p95=0.22,
        tail_policy="UPPER_TAIL",
    )
    assert ref.metric_name == "motion_divergence"

    # Sample count < 100 rejected
    with pytest.raises(ValueError, match="Insufficient reference sample count"):
        BaselineReference(
            metric_name="test_insufficient",
            feature_version="1.0",
            dataset_name="FaceForensics++",
            sample_video_count=99,
            mean=0.15,
            std=0.03,
            median=0.14,
            p05=0.10,
            p95=0.22,
            tail_policy="UPPER_TAIL",
        )

    # Inverted quantile ordering (p05 > median or median > p95)
    with pytest.raises(ValueError, match="Invalid quantile ordering"):
        BaselineReference(
            metric_name="test_inverted",
            feature_version="1.0",
            dataset_name="FaceForensics++",
            sample_video_count=100,
            mean=0.15,
            std=0.03,
            median=0.14,
            p05=0.18,  # p05 > median!
            p95=0.22,
            tail_policy="UPPER_TAIL",
        )


def test_empirical_baseline_manager_loading_and_evaluation(tmp_path: Path):
    """Verify EmpiricalBaselineManager loads valid JSON and evaluates sample percentiles."""
    baseline_file = tmp_path / "valid_baseline.json"
    data = {
        "schema_version": "2.0",
        "reference_dataset": "FaceForensics++ C23",
        "metrics": {
            "spatial_laplacian_var": {
                "feature_version": "1.0",
                "dataset_name": "FaceForensics++ C23",
                "sample_video_count": 150,
                "mean": 0.30,
                "std": 0.05,
                "median": 0.29,
                "p05": 0.20,
                "p95": 0.42,
                "tail_policy": "TWO_TAILED",
            },
            "landmark_velocity_jitter": {
                "feature_version": "1.0",
                "dataset_name": "FaceForensics++ C23",
                "sample_video_count": 105,
                "mean": 0.10,
                "std": 0.02,
                "median": 0.09,
                "p05": 0.06,
                "p95": 0.16,
                "tail_policy": "UPPER_TAIL",
            },
        },
    }
    baseline_file.write_text(json.dumps(data), encoding="utf-8")

    mgr = EmpiricalBaselineManager(baseline_file)
    assert len(mgr.baselines) == 2

    # Within bounds (not anomaly)
    res_normal = mgr.evaluate_sample("spatial_laplacian_var", 0.31)
    assert res_normal["is_calibrated"] is True
    assert res_normal["is_anomaly"] is False
    assert res_normal["z_score"] is not None

    # Outside upper bound (anomaly)
    res_high = mgr.evaluate_sample("landmark_velocity_jitter", 0.25)
    assert res_high["is_calibrated"] is True
    assert res_high["is_anomaly"] is True

    # Missing metric returns uncalibrated state
    res_missing = mgr.evaluate_sample("nonexistent_metric", 0.5)
    assert res_missing["is_calibrated"] is False
    assert res_missing["is_anomaly"] is False


def test_corrupt_baseline_file_handling(tmp_path: Path):
    """Verify that corrupt JSON or missing schema raises clear exceptions."""
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{invalid json", encoding="utf-8")

    mgr = EmpiricalBaselineManager()
    with pytest.raises(ValueError, match="Corrupt or invalid JSON baseline file"):
        mgr.load_baselines(corrupt_file)

    missing_schema = tmp_path / "missing_schema.json"
    missing_schema.write_text(json.dumps({"some_key": 123}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required 'schema_version' or 'metrics' key"):
        mgr.load_baselines(missing_schema)
