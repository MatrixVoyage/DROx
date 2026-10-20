"""Tests for Forensic Decision Engine, coverage verification, target scores, and confound handling."""
import pytest
import torch
import numpy as np

from src.forensics.schema import (
    ModuleExecutionStatus,
    ConfoundCode,
    EvidenceDependency,
    ManipulationAssessment,
)
from src.forensics.spatial_temporal_pipeline import (
    has_sufficient_negative_coverage,
    select_target_score,
)


def test_has_sufficient_negative_coverage():
    """Verify that negative assessment requires COMPLETED status across all required modules."""
    required = ["m13_spatial", "m18_kinematics"]

    # All completed
    statuses_ok = {
        "m13_spatial": ModuleExecutionStatus.COMPLETED,
        "m18_kinematics": ModuleExecutionStatus.COMPLETED,
    }
    assert has_sufficient_negative_coverage(statuses_ok, required) is True

    # One failed
    statuses_failed = {
        "m13_spatial": ModuleExecutionStatus.COMPLETED,
        "m18_kinematics": ModuleExecutionStatus.FAILED,
    }
    assert has_sufficient_negative_coverage(statuses_failed, required) is False

    # One uncalibrated
    statuses_uncalibrated = {
        "m13_spatial": ModuleExecutionStatus.COMPLETED,
        "m18_kinematics": ModuleExecutionStatus.UNCALIBRATED,
    }
    assert has_sufficient_negative_coverage(statuses_uncalibrated, required) is False

    # One missing entirely
    statuses_missing = {
        "m13_spatial": ModuleExecutionStatus.COMPLETED,
    }
    assert has_sufficient_negative_coverage(statuses_missing, required) is False


def test_select_target_score():
    """Verify target score adapter handles single-logit and two-logit shapes safely."""
    # Single-logit positive class (class 1)
    single_logit = torch.tensor([2.5])
    score_pos = select_target_score(single_logit, target_class_idx=1, binary_mode="single_logit_positive")
    assert pytest.approx(score_pos.item(), 0.001) == 2.5

    # Single-logit negative class (class 0)
    score_neg = select_target_score(single_logit, target_class_idx=0, binary_mode="single_logit_positive")
    assert pytest.approx(score_neg.item(), 0.001) == -2.5

    # Two-logit model: batch of 2, 2 classes
    two_logit = torch.tensor([[1.2, 3.4], [-0.5, 0.8]])
    score_class1 = select_target_score(two_logit, target_class_idx=1)
    assert pytest.approx(score_class1.item(), 0.001) == 3.4 + 0.8

    score_class0 = select_target_score(two_logit, target_class_idx=0)
    assert pytest.approx(score_class0.item(), 0.001) == 1.2 + (-0.5)

    # Out of bounds class index
    with pytest.raises(ValueError, match="outside model output range"):
        select_target_score(two_logit, target_class_idx=5)

    # Invalid dimension
    with pytest.raises(ValueError, match="Unexpected logits shape"):
        select_target_score(torch.zeros((1, 2, 3)), target_class_idx=0)


def test_evidence_dependency_and_confound_codes():
    """Verify structured confound codes and dependency groupings prevent artificial corroboration."""
    dep_spatial = EvidenceDependency(
        finding_id="FIND-101",
        modality="wiener_filter",
        dependency_group="SPATIAL_TEXTURE",
        calibration_id="CAL-V24",
        quality_flags=(ConfoundCode.VIDEO_COMPRESSION,),
    )
    dep_laplacian = EvidenceDependency(
        finding_id="FIND-102",
        modality="laplacian_highfreq",
        dependency_group="SPATIAL_TEXTURE",
        calibration_id="CAL-V24",
        quality_flags=(ConfoundCode.VIDEO_COMPRESSION,),
    )
    # Both share the SPATIAL_TEXTURE dependency group
    assert dep_spatial.dependency_group == dep_laplacian.dependency_group
    assert ConfoundCode.VIDEO_COMPRESSION == "video_compression"
