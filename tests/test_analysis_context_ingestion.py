"""Tests for AnalysisContext dual-pathway ingestion, metadata extraction, and schema serialization."""
from pathlib import Path
import hashlib
import numpy as np
import pytest
import cv2
import torch

from src.forensics.schema import (
    FindingType,
    EvidenceSeverity,
    ModalityGroup,
    ManipulationAssessment,
    ProvenanceAssessment,
    ModuleExecutionStatus,
    MeasurementDefinition,
    BaselineComparison,
    EvidenceFinding,
    ForensicEvidenceReport,
)
from src.forensics.context import (
    FrameReference,
    TemporalWindow,
    VideoMetadata,
    AnalysisContext,
)


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    """Creates a temporary 30-frame synthetic test video."""
    video_path = tmp_path / "test_synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 25.0
    w, h = 320, 240
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    for i in range(30):
        # Create a frame with some content
        img = np.full((h, w, 3), (i * 8) % 256, dtype=np.uint8)
        cv2.putText(img, f"Frame {i}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out.write(img)
    out.release()
    return video_path


def test_chunked_sha256_matches_full_hash(dummy_video: Path):
    """Verify that chunked 64 KB hashing exactly matches hashlib full file read."""
    # Full read
    expected_sha = hashlib.sha256(dummy_video.read_bytes()).hexdigest()

    ctx = AnalysisContext.build_from_video(
        dummy_video,
        spatial_sample_count=8,
        temporal_window_length=8,
        max_temporal_windows=2,
    )
    assert ctx.file_sha256 == expected_sha
    assert ctx.file_size_bytes == dummy_video.stat().st_size


def test_dual_pathway_sampling(dummy_video: Path):
    """Verify spatial sparse sampling and temporal dense sequential windows."""
    ctx = AnalysisContext.build_from_video(
        dummy_video,
        spatial_sample_count=10,
        temporal_window_length=8,
        max_temporal_windows=2,
    )

    # 1. Spatial pathway
    assert len(ctx.spatial_frames) == 10
    for idx, ref in enumerate(ctx.spatial_frames):
        assert isinstance(ref, FrameReference)
        assert ref.sample_idx == idx
        assert ref.image_bgr.shape == (240, 320, 3)
        assert ref.pts_sec >= 0.0

    # 2. Temporal pathway: 2 windows of 8 consecutive frames
    assert len(ctx.temporal_windows) == 2
    for w_idx, win in enumerate(ctx.temporal_windows):
        assert isinstance(win, TemporalWindow)
        assert win.window_idx == w_idx
        assert len(win.frames) == 8
        assert win.start_time_sec <= win.end_time_sec
        # Check consecutive continuity
        for k in range(len(win.frames) - 1):
            assert win.frames[k + 1].pts_sec >= win.frames[k].pts_sec


def test_schema_serialization():
    """Verify that ForensicEvidenceReport.to_dict handles Enum, NumPy, PyTorch, and Path objects."""
    comparison = BaselineComparison(
        baseline_version="v24.1",
        metric_name="landmark_velocity_jitter",
        sample_value=np.float32(0.85),
        baseline_n=120,
        baseline_mean=0.20,
        baseline_std=0.05,
        z_score=np.float64(13.0),
        p_value=0.0001,
        is_anomaly=True,
        is_calibrated=True,
    )

    finding = EvidenceFinding(
        finding_id="FIND-001",
        module="m18_kinematics",
        modality_group=ModalityGroup.FACIAL_KINEMATICS,
        finding_type=FindingType.VALIDATED_FORENSIC_EVIDENCE,
        severity=EvidenceSeverity.HIGH,
        title="Abnormal Landmark Velocity Variance",
        description="Spatiotemporal velocity exceeds authentic baseline distribution.",
        numeric_value=np.float32(0.85),
        baseline_comparison=comparison,
        start_time_sec=0.0,
        end_time_sec=0.64,
        frame_indices=[0, 1, 2, 3],
        visual_artifact_uri="reports/artifacts/kinematic_traj.png",
        alternative_explanations=["Rapid head movement"],
        quality_flags=[],
    )

    report = ForensicEvidenceReport(
        report_id="RPT-TEST001",
        video_filename="test_synthetic.mp4",
        file_sha256="deadbeef" * 8,
        container_duration_sec=1.2,
        analyzed_duration_sec=1.2,
        total_frames_analyzed=30,
        manipulation_assessment=ManipulationAssessment.SUPPORTING_MANIPULATION,
        provenance_assessment=ProvenanceAssessment.NO_CREDENTIAL_FOUND,
        executive_summary="Evidence supporting manipulation detected.",
        findings=[finding],
        module_statuses={"m18_kinematics": ModuleExecutionStatus.COMPLETED},
        quality_assessment={"flags": []},
        limitations_and_caveats=["Test caveat"],
        provenance_details={},
        reproducibility_metadata={"tensor_sample": torch.tensor([1.0, 2.0])},
    )

    d = report.to_dict()
    assert isinstance(d, dict)
    assert d["manipulation_assessment"] == "SUPPORTING_MANIPULATION"
    assert d["provenance_assessment"] == "NO_CREDENTIAL_FOUND"
    assert d["findings"][0]["severity"] == "HIGH"
    assert d["findings"][0]["modality_group"] == "FACIAL_KINEMATICS"
    assert isinstance(d["findings"][0]["numeric_value"], float)
    assert d["reproducibility_metadata"]["tensor_sample"] == [1.0, 2.0]
