"""Milestone 23: Tests for multi-model API serving and cross-engine comparison endpoint."""

import io
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, set_predictor, get_predictor, AVAILABLE_ENGINES
from src.inference.predictor import PredictionResult, VideoPredictor, EnsemblePredictor


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


def test_health_endpoint_m23(client):
    """Verify GET /health returns updated version, champion engine, and available engines."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["version"] == "2.1.0"
    assert data["champion_engine"] == "v5-family-invariant"
    assert data["ensemble_engine"] == "m22-ensemble"
    assert "m22-ensemble" in data["available_engines"]
    assert "m18-dual-stream" in data["available_engines"]
    assert "m17-sensor-residuals" in data["available_engines"]


def test_model_info_m22_ensemble(client):
    """Verify GET /api/v1/model-info for m22-ensemble contains provenance and parameters."""
    resp = client.get("/api/v1/model-info?engine=m22-ensemble")
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine"] == "m22-ensemble"
    assert "M22 Log-Odds Fusion" in data["display_name"]
    assert "Platt Fusion" in data["architecture"]
    assert data["default_threshold"] == 0.9191
    prov = data.get("checkpoint_provenance")
    assert prov is not None
    assert prov["model_id"] == "m22-ensemble"
    assert prov["benchmark_results"]["face2face_auc"] == 0.8118
    assert prov["parameters"]["w1_m13"] == 0.4092
    assert prov["parameters"]["w2_m18"] == 0.3590
    assert prov["parameters"]["tau_95"] == 0.9191


def test_model_info_m18_dual_stream(client):
    """Verify GET /api/v1/model-info for m18-dual-stream contains kinematics details."""
    resp = client.get("/api/v1/model-info?engine=m18-dual-stream")
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine"] == "m18-dual-stream"
    assert "Kinematics" in data["display_name"]
    assert "Kinematics" in data["architecture"]
    prov = data.get("checkpoint_provenance")
    assert prov is not None
    assert prov["model_id"] == "m18-dual-stream"
    assert prov["benchmark_results"]["face2face_auc"] == 0.7858


def test_model_info_m17_sensor_residuals(client):
    """Verify GET /api/v1/model-info for m17-sensor-residuals."""
    resp = client.get("/api/v1/model-info?engine=m17-sensor-residuals")
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine"] == "m17-sensor-residuals"
    assert "Residuals" in data["architecture"]


def test_predict_m22_ensemble_mock(client):
    """Verify POST /api/v1/predict with engine=m22-ensemble handles inference correctly."""
    mock_pred = MagicMock(spec=VideoPredictor)
    mock_result = PredictionResult(
        video_path="sample.mp4",
        verdict="Deepfake",
        probability=0.9412,
        raw_logit=2.77,
        confidence=0.9412,
        threshold=0.9191,
        frames_analyzed=16,
        faces_detected=16,
        device="cpu",
        model_version="m22-ensemble",
        latency_ms={"total_ms": 185.0},
        decision="FAKE",
        metadata={
            "ensemble": True,
            "signals": {
                "m13_5_probability": 0.88,
                "m18_probability": 0.85,
                "fused_probability": 0.9412,
                "static_freq_score": 0.72,
                "static_texture_score": 0.65,
                "kinematic_motion_score": 0.81,
            },
        },
    )
    mock_pred.predict_video.return_value = mock_result
    set_predictor(mock_pred)

    try:
        fake_video = io.BytesIO(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00")
        resp = client.post(
            "/api/v1/predict",
            files={"file": ("sample.mp4", fake_video, "video/mp4")},
            data={"engine": "m22-ensemble", "threshold": 0.9191},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "Deepfake"
        assert data["probability"] == 0.9412
        assert data["decision"] == "FAKE"
        assert data["engine"] == "m22-ensemble"
        assert "signals" in data["metadata"]
        assert data["metadata"]["signals"]["fused_probability"] == 0.9412
    finally:
        set_predictor(None)


def test_compare_endpoint_concordant(client):
    """Verify POST /api/v1/compare returns side-by-side results and consensus."""
    mock_pred = MagicMock(spec=VideoPredictor)
    mock_result = PredictionResult(
        video_path="sample.mp4",
        verdict="Deepfake",
        probability=0.92,
        raw_logit=2.4,
        confidence=0.92,
        threshold=0.5,
        frames_analyzed=16,
        faces_detected=16,
        device="cpu",
        model_version="test",
        latency_ms={"total_ms": 100.0},
        decision="FAKE",
    )
    mock_pred.predict_video.return_value = mock_result
    set_predictor(mock_pred)

    try:
        fake_video = io.BytesIO(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00")
        resp = client.post(
            "/api/v1/compare",
            files={"file": ("test_clip.mp4", fake_video, "video/mp4")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["video_filename"] == "test_clip.mp4"
        assert data["consensus"] == "CONCORDANT_FAKE"
        assert "engines" in data
        assert "v5-family-invariant" in data["engines"]
        assert "m18-dual-stream" in data["engines"]
        assert "m22-ensemble" in data["engines"]
        assert "signals" in data
        assert data["signals"]["m22_fused_probability"] == 0.92
    finally:
        set_predictor(None)


def test_forensic_audit_invalid_extension(client):
    """Verify POST /api/v1/forensic-audit rejects invalid file extensions with 400."""
    fake_txt = io.BytesIO(b"not a video")
    resp = client.post(
        "/api/v1/forensic-audit",
        files={"file": ("test.txt", fake_txt, "text/plain")},
    )
    assert resp.status_code == 400
    assert "Unsupported file format" in resp.json()["detail"]


def test_forensic_audit_endpoint_success(client, tmp_path):
    """Verify POST /api/v1/forensic-audit successfully processes a video."""
    import cv2
    import numpy as np
    from src.api.app import set_forensic_engine
    from src.forensics.spatial_temporal_pipeline import SpatialTemporalForensicEngine

    # Create small dummy synthetic video
    vid_path = tmp_path / "api_test_synth.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 25.0, (160, 120))
    for i in range(16):
        frame = np.full((120, 160, 3), (i * 15) % 256, dtype=np.uint8)
        out.write(frame)
    out.release()

    engine = SpatialTemporalForensicEngine(device="cpu")
    set_forensic_engine(engine)

    try:
        with open(vid_path, "rb") as f:
            resp = client.post(
                "/api/v1/forensic-audit",
                files={"file": ("api_test_synth.mp4", f, "video/mp4")},
                data={"windows": "2", "frames_per_window": "8", "window_duration": "1.0"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["video_filename"] == "api_test_synth.mp4"
        assert data["verdict"] in ["REAL", "DEEPFAKE", "INCONCLUSIVE"]
        assert "saliency_heatmap_base64" in data
        assert len(data["saliency_heatmap_base64"]) > 0
        assert "windows_analyzed" in data
        assert "provenance" in data
        assert "video_fake_probability" in data
    finally:
        set_forensic_engine(None)

