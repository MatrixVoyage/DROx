"""Unit and integration tests for Video-Specific Forensic Observations and Telemetry."""
from pathlib import Path
import json
import numpy as np
import pytest
import cv2

from src.forensics.loader import ForensicVideoLoader
from src.forensics.context import (
    ForensicAnalysisConfig,
    ForensicVideoReport,
)
from src.forensics.schema import (
    VideoSpecificTelemetry,
    StreamTelemetry,
    SpatialTelemetry,
    TemporalTelemetry,
    BaselineTelemetry,
    QualityAndConfounderTelemetry,
    ForensicDecisionTelemetry,
    format_telemetry_narrative,
)
from src.forensics.spatial_temporal_pipeline import SpatialTemporalForensicEngine
from src.forensics.baseline import EmpiricalBaselineManager


def _create_synthetic_video(
    path: Path,
    num_frames: int = 24,
    width: int = 320,
    height: int = 240,
    fps: float = 24.0,
    with_face: bool = True,
    motion_speed: int = 2,
    brightness: int = 200,
) -> Path:
    """Helper to generate deterministic synthetic test videos with distinct properties."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for i in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        if with_face:
            x = 40 + i * motion_speed
            y = 50
            # Face box
            cv2.rectangle(img, (x, y), (x + 80, y + 80), (brightness, brightness, brightness), -1)
            # Eyes and mouth
            cv2.circle(img, (x + 25, y + 25), 5, (20, 20, 20), -1)
            cv2.circle(img, (x + 55, y + 25), 5, (20, 20, 20), -1)
            cv2.rectangle(img, (x + 25, y + 55), (x + 55, y + 65), (20, 20, 20), -1)
        out.write(img)
    out.release()
    return path


def test_different_input_videos_produce_distinct_telemetry(tmp_path: Path):
    """Proves two different input videos produce telemetry derived from their respective analysis."""
    video_a = _create_synthetic_video(
        tmp_path / "vid_a.mp4",
        num_frames=20,
        width=320,
        height=240,
        fps=20.0,
        motion_speed=1,
        brightness=180,
    )
    video_b = _create_synthetic_video(
        tmp_path / "vid_b.mp4",
        num_frames=36,
        width=400,
        height=300,
        fps=30.0,
        motion_speed=4,
        brightness=240,
    )

    cfg = ForensicAnalysisConfig(frames_per_window=8, window_duration_sec=1.0, max_windows=2)
    engine = SpatialTemporalForensicEngine(config=cfg, device="cpu")

    report_a = engine.analyze_video(video_a)
    report_b = engine.analyze_video(video_b)

    st_a = report_a.structured_telemetry
    st_b = report_b.structured_telemetry

    assert st_a is not None and st_b is not None

    # 1. Stream Telemetry must be distinctly derived from input files
    assert st_a["stream"]["source_file_name"] == "vid_a.mp4"
    assert st_b["stream"]["source_file_name"] == "vid_b.mp4"
    assert st_a["stream"]["file_sha256"] != st_b["stream"]["file_sha256"]
    assert st_a["stream"]["width"] == 320
    assert st_b["stream"]["width"] == 400
    assert st_a["stream"]["height"] == 240
    assert st_b["stream"]["height"] == 300
    assert st_a["stream"]["container_fps"] == pytest.approx(20.0, abs=0.5)
    assert st_b["stream"]["container_fps"] == pytest.approx(30.0, abs=0.5)

    # 2. Temporal frame counts and differential measurements must reflect actual content
    assert st_a["temporal"]["inter_frame_differential_energy"] != st_b["temporal"]["inter_frame_differential_energy"]

    # 3. Explanations must contain their respective filenames and measurements
    assert "vid_a.mp4" in report_a.forensic_explanation
    assert "vid_b.mp4" in report_b.forensic_explanation
    assert st_a["stream"]["file_sha256"][:16] in report_a.forensic_explanation
    assert st_b["stream"]["file_sha256"][:16] in report_b.forensic_explanation


def test_displayed_narrative_matches_underlying_structured_telemetry(tmp_path: Path):
    """Proves displayed values match the underlying structured measurements without discrepancy."""
    video = _create_synthetic_video(tmp_path / "measured_video.mp4", num_frames=24)
    cfg = ForensicAnalysisConfig(frames_per_window=8, window_duration_sec=1.0, max_windows=2)
    engine = SpatialTemporalForensicEngine(config=cfg, device="cpu")

    report = engine.analyze_video(video)
    st = report.structured_telemetry
    assert st is not None

    narrative = report.forensic_explanation

    # Check Stream section consistency
    assert f"Source File: {st['stream']['source_file_name']}" in narrative
    assert f"{st['stream']['file_sha256'][:20]}" in narrative
    assert f"{st['stream']['width']}x{st['stream']['height']}" in narrative

    # Check Spatial section consistency
    outer_e = f"{st['spatial']['outer_boundary_energy']:.4f}"
    ratio_str = f"{st['spatial']['boundary_to_landmark_ratio']:.2f}"
    assert outer_e in narrative
    assert ratio_str in narrative

    # Check Temporal section consistency
    diff_rate_str = f"{st['temporal']['inter_frame_differential_energy']:.2f}"
    assert diff_rate_str in narrative
    peak_w = st["temporal"]["peak_window"]
    assert f"Window #{peak_w['window_id']}" in narrative
    assert f"{peak_w['start_sec']:.1f}s-{peak_w['end_sec']:.1f}s" in narrative

    # Check Decision section consistency
    assert f"Assessment State: {st['decision']['decision_state']}" in narrative
    assert f"Active Threshold {st['decision']['active_threshold']:.2f}" in narrative


def test_peak_temporal_window_mapping(tmp_path: Path):
    """Proves peak window detail correctly maps to the maximum calibrated score interval."""
    video = _create_synthetic_video(tmp_path / "peak_test.mp4", num_frames=32)
    cfg = ForensicAnalysisConfig(frames_per_window=8, window_duration_sec=1.0, max_windows=4)
    engine = SpatialTemporalForensicEngine(config=cfg, device="cpu")

    report = engine.analyze_video(video)
    st = report.structured_telemetry
    peak = st["temporal"]["peak_window"]

    # Locate maximum window in report.windows_analyzed
    max_w = max(report.windows_analyzed, key=lambda w: w.calibrated_fake_probability)
    assert peak["window_id"] == max_w.window_id
    assert peak["start_sec"] == pytest.approx(max_w.start_sec, abs=1e-3)
    assert peak["end_sec"] == pytest.approx(max_w.end_sec, abs=1e-3)
    assert peak["calibrated_fake_probability"] == pytest.approx(max_w.calibrated_fake_probability, abs=1e-4)
    assert peak["selection_method"] == "Highest calibrated manipulation probability among evaluated temporal windows"


def test_baseline_guards_prevent_fabricated_deviations(tmp_path: Path):
    """Proves an unavailable baseline never produces fabricated z-scores or deviations."""
    video = _create_synthetic_video(tmp_path / "uncalibrated_baseline.mp4", num_frames=16)

    # Engine initialized with no baseline distributions
    empty_mgr = EmpiricalBaselineManager()
    engine = SpatialTemporalForensicEngine(baseline_manager=empty_mgr, device="cpu")

    report = engine.analyze_video(video)
    baseline_st = report.structured_telemetry["baseline"]

    assert baseline_st["is_eligible"] is False
    assert baseline_st["kinematic_z_score"] is None
    assert baseline_st["spatial_ratio_z_score"] is None
    assert "UNAVAILABLE" in report.forensic_explanation
    assert "Standardized z-scores omitted" in report.forensic_explanation


def test_no_unsupported_manipulation_attribution(tmp_path: Path):
    """Proves the pipeline does not emit unvalidated multi-class attributions (e.g. Face2Face)."""
    video = _create_synthetic_video(tmp_path / "attribution_guard.mp4", num_frames=16)
    engine = SpatialTemporalForensicEngine(device="cpu")

    report = engine.analyze_video(video)
    st = report.structured_telemetry

    attribution = st["decision"]["attribution"]
    family = report.detected_manipulation_family
    # Must NOT claim Face2Face, FaceSwap, NeuralTextures without a dedicated attribution head
    assert attribution not in ["Face2Face", "FaceSwap", "NeuralTextures", "Deepfakes"]
    assert family not in ["Face2Face", "FaceSwap", "NeuralTextures", "Deepfakes"]
    assert attribution in ["Unspecified / Binary Detection Only", "None"]
    assert family in ["Unspecified / Binary Detection Only", "None"]


def test_failure_handling_and_inconclusive_verdict(tmp_path: Path):
    """Proves missing face coverage or module failures result in INCONCLUSIVE, not a negative verdict."""
    # Create pure black blank video without faces
    blank_video = _create_synthetic_video(tmp_path / "blank_video.mp4", num_frames=16, with_face=False)
    engine = SpatialTemporalForensicEngine(device="cpu")

    report = engine.analyze_video(blank_video)
    st = report.structured_telemetry

    assert report.verdict == "INCONCLUSIVE"
    assert st["quality"]["sufficient_coverage"] is False
    assert st["quality"]["face_detected_any"] is False
    assert any("tracking_instability" in flag for flag in st["quality"]["confound_codes"])
    # Confirm a video without face detections cannot be certified as 'REAL'
    assert report.verdict != "REAL"


def test_determinism_and_repeatability(tmp_path: Path):
    """Proves repeated analysis with identical inputs and configuration produces consistent findings."""
    video = _create_synthetic_video(tmp_path / "repeat_video.mp4", num_frames=24)
    cfg = ForensicAnalysisConfig(frames_per_window=8, window_duration_sec=1.0, max_windows=2)
    engine = SpatialTemporalForensicEngine(config=cfg, device="cpu")

    rep1 = engine.analyze_video(video)
    rep2 = engine.analyze_video(video)

    st1 = rep1.structured_telemetry
    st2 = rep2.structured_telemetry

    assert st1["stream"]["file_sha256"] == st2["stream"]["file_sha256"]
    assert st1["spatial"]["outer_boundary_energy"] == pytest.approx(
        st2["spatial"]["outer_boundary_energy"], abs=1e-4
    )
    assert st1["temporal"]["inter_frame_differential_energy"] == pytest.approx(
        st2["temporal"]["inter_frame_differential_energy"], abs=1e-4
    )
    assert st1["decision"]["calibrated_probability"] == pytest.approx(
        st2["decision"]["calibrated_probability"], abs=1e-4
    )
    assert rep1.verdict == rep2.verdict


def test_api_forensic_audit_endpoint_structured_telemetry(tmp_path: Path):
    """Proves the FastAPI /api/v1/forensic-audit route returns structured telemetry with all 6 sections."""
    from fastapi.testclient import TestClient
    from src.api.app import app

    video = _create_synthetic_video(tmp_path / "api_test_vid.mp4", num_frames=16)

    client = TestClient(app)
    with open(video, "rb") as f:
        response = client.post(
            "/api/v1/forensic-audit",
            files={"file": ("api_test_vid.mp4", f, "video/mp4")},
            params={"windows": 2, "window_duration": 1.0, "frames_per_window": 8},
        )

    assert response.status_code == 200
    data = response.json()

    assert "structured_telemetry" in data
    st = data["structured_telemetry"]
    assert st is not None
    assert st["schema_version"] == "2.0.0"

    # Verify all 6 core sections exist
    assert "stream" in st
    assert "spatial" in st
    assert "temporal" in st
    assert "baseline" in st
    assert "quality" in st
    assert "decision" in st

    # Verify video-specific properties
    assert st["stream"]["source_file_name"] == "api_test_vid.mp4"
    assert len(st["stream"]["file_sha256"].replace("sha256:", "")) == 64
    assert st["stream"]["width"] == 320
    assert st["stream"]["height"] == 240
    assert "forensic_explanation" in data
    assert "Source File: api_test_vid.mp4" in data["forensic_explanation"]

