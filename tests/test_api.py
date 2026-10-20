"""Unit and integration tests for FastAPI inference endpoints."""

import io
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, set_predictor
from src.inference.predictor import PredictionResult, VideoPredictor


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Verify GET /health returns 200 and expected schema."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model_loaded" in data


def test_model_info_endpoint(client):
    """Verify GET /api/v1/model-info returns architecture details and limitations."""
    resp = client.get("/api/v1/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert "architecture" in data
    assert "backbone" in data
    assert "reference_paper" in data
    assert "research_limitations" in data
    assert len(data["research_limitations"]) > 0
    assert data["sequence_length"] == 16


def test_predict_invalid_extension(client):
    """Verify POST /api/v1/predict rejects non-video files with 400."""
    fake_file = io.BytesIO(b"dummy non-video text content")
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("test.txt", fake_file, "text/plain")},
    )
    assert resp.status_code == 400
    assert "Unsupported file format" in resp.json()["detail"]


def test_predict_mock_success(client):
    """Verify POST /api/v1/predict returns structured PredictionResult."""
    mock_pred = MagicMock(spec=VideoPredictor)
    mock_result = PredictionResult(
        video_path="temp.mp4",
        verdict="Real",
        probability=0.23,
        raw_logit=-1.2,
        confidence=0.77,
        threshold=0.5,
        frames_analyzed=16,
        faces_detected=16,
        device="cpu",
        model_version="best.pth",
        latency_ms={"preprocessing_ms": 120.0, "model_forward_ms": 20.0, "total_ms": 140.0},
    )
    mock_pred.predict_video.return_value = mock_result
    set_predictor(mock_pred)

    fake_video = io.BytesIO(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00")
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("sample.mp4", fake_video, "video/mp4")},
        data={"threshold": 0.5},
    )

    set_predictor(None)  # Reset
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "Real"
    assert data["probability"] == 0.23
    assert data["faces_detected"] == 16
    assert data["original_filename"] == "sample.mp4"
    assert data["engine"] == "v1-baseline"


def test_dual_engine_selection(client):
    """Verify dual engine queries for v1-baseline and v2-evolved."""
    resp_base = client.get("/api/v1/model-info?engine=v1-baseline")
    assert resp_base.status_code == 200
    data_base = resp_base.json()
    assert data_base["engine"] == "v1-baseline"
    assert "v1-baseline" in data_base["available_engines"]
    assert "v2-evolved" in data_base["available_engines"]

    resp_evo = client.get("/api/v1/model-info?engine=v2-evolved")
    assert resp_evo.status_code == 200
    data_evo = resp_evo.json()
    assert data_evo["engine"] == "v2-evolved"
    assert "efficientnet" in data_evo["backbone"].lower()
    assert "transformer" in data_evo["temporal_module"].lower()

    # Verify invalid engine is rejected with 400
    fake_video = io.BytesIO(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00")
    resp_inv = client.post(
        "/api/v1/predict",
        files={"file": ("sample.mp4", fake_video, "video/mp4")},
        data={"engine": "non-existent-engine"},
    )
    assert resp_inv.status_code == 400
    assert "Unsupported engine" in resp_inv.json()["detail"]
