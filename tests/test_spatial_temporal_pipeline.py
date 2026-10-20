"""Tests for SpatialTemporalForensicEngine and ForensicVideoLoader."""
from pathlib import Path
import json
import numpy as np
import pytest
import cv2
import torch

from src.forensics.loader import ForensicVideoLoader
from src.forensics.context import (
    ForensicAnalysisConfig,
    ForensicVideoReport,
    FrameReference,
    TemporalWindow,
)
from src.forensics.spatial_temporal_pipeline import (
    SpatialTemporalForensicEngine,
    DEFAULT_RAW_THRESHOLD,
    DEFAULT_PLATT_THRESHOLD,
)


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Creates a temporary 30-frame synthetic test video with a moving face-like target."""
    video_path = tmp_path / "test_synth_forensic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 25.0
    w, h = 320, 240
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    for i in range(30):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Draw moving face box
        x = 50 + i * 2
        y = 60
        cv2.rectangle(img, (x, y), (x + 80, y + 80), (200, 220, 240), -1)
        # Inner facial feature simulation
        cv2.circle(img, (x + 25, y + 25), 6, (40, 40, 40), -1)
        cv2.circle(img, (x + 55, y + 25), 6, (40, 40, 40), -1)
        cv2.ellipse(img, (x + 40, y + 55), (15, 8), 0, 0, 180, (40, 40, 40), 2)
        out.write(img)
    out.release()
    return video_path


def test_video_loader_inspect_and_load(synthetic_video: Path):
    loader = ForensicVideoLoader()
    meta = loader.inspect_container(synthetic_video)

    assert meta["duration_sec"] > 0
    assert meta["fps"] > 0
    assert meta["width"] == 320
    assert meta["height"] == 240

    frames = loader.load_temporal_window_frames(
        synthetic_video,
        start_sec=0.0,
        duration_sec=1.0,
        target_frame_count=8,
    )
    assert len(frames) == 8
    assert isinstance(frames[0], FrameReference)
    assert frames[0].image_bgr.shape == (240, 320, 3)


def test_spatial_temporal_engine_calibration():
    engine = SpatialTemporalForensicEngine(device="cpu")
    # Test probability clipping and Platt scaling
    prob_low = engine.calibrate_probability(0.05)
    prob_mid = engine.calibrate_probability(0.50)
    prob_high = engine.calibrate_probability(0.95)

    assert 0.0 <= prob_low <= 1.0
    assert 0.0 <= prob_mid <= 1.0
    assert 0.0 <= prob_high <= 1.0
    assert prob_low < prob_mid < prob_high


def test_spatial_saliency_and_kinematic_energy():
    engine = SpatialTemporalForensicEngine(device="cpu")
    
    crops = [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8) for _ in range(5)]
    saliency = engine._compute_spatial_saliency(crops)
    assert saliency.shape == (112, 112)
    assert 0.0 <= float(saliency.min())
    assert float(saliency.max()) <= 1.0

    energy = engine._compute_kinematic_energy(crops)
    assert isinstance(energy, float)
    assert energy >= 0.0


def test_analyze_window_on_synthetic_frames():
    engine = SpatialTemporalForensicEngine(device="cpu")

    frames = []
    for i in range(16):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        x = 50 + i * 2
        y = 60
        cv2.rectangle(img, (x, y), (x + 80, y + 80), (220, 220, 220), -1)
        frames.append(FrameReference(
            sample_idx=i,
            original_frame_idx=i,
            pts_sec=i * 0.04,
            image_bgr=img,
        ))

    window = TemporalWindow(
        window_id=0,
        start_sec=0.0,
        end_sec=0.64,
        target_frames=16,
    )

    analysis = engine.analyze_window(frames, window)
    assert analysis.window_id == 0
    assert 0.0 <= analysis.calibrated_fake_probability <= 1.0
    assert analysis.spatial_saliency.shape == (112, 112)
    assert analysis.kinematic_residual_energy >= 0.0
    assert len(analysis.frame_anomaly_scores) == 16


def test_end_to_end_analyze_video(synthetic_video: Path, tmp_path: Path):
    cfg = ForensicAnalysisConfig(
        target_face_size=112,
        frames_per_window=8,
        window_duration_sec=1.0,
        max_windows=2,
    )
    engine = SpatialTemporalForensicEngine(config=cfg, device="cpu")

    report = engine.analyze_video(synthetic_video)
    assert isinstance(report, ForensicVideoReport)
    assert report.verdict in ["REAL", "DEEPFAKE", "INCONCLUSIVE"]
    assert 0.0 <= report.video_fake_probability <= 1.0
    assert len(report.confidence_interval) == 2
    assert report.confidence_interval[0] <= report.confidence_interval[1]
    assert len(report.windows_analyzed) >= 1
    assert report.overall_spatial_saliency.shape == (112, 112)

    # Test JSON export
    json_path = tmp_path / "forensic_report.json"
    exported_path = engine.export_report_json(report, json_path)
    assert Path(exported_path).exists()

    with open(exported_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["verdict"] == report.verdict
    assert "video_fake_probability" in data
    assert "confidence_interval" in data
    assert "windows_analyzed" in data
    assert "provenance" in data


def test_cli_forensic_audit_subcommand(synthetic_video: Path, tmp_path: Path):
    from src.cli import main
    import sys

    out_file = tmp_path / "cli_forensic_output.json"
    test_args = [
        "cli.py",
        "forensic-audit",
        "--video", str(synthetic_video),
        "--windows", "2",
        "--frames-per-window", "8",
        "--cpu",
        "--output", str(out_file),
    ]

    orig_argv = sys.argv
    try:
        sys.argv = test_args
        main()
    finally:
        sys.argv = orig_argv

    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "verdict" in data
    assert "video_fake_probability" in data

