"""Tests for Calibrated 3-State Rejection REST API Endpoint (M12.4)."""

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


def test_v3_robust_model_info(client):
    """Verify GET /api/v1/model-info for v3-robust (M12)."""
    resp = client.get("/api/v1/model-info?engine=v3-robust")
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine"] == "v3-robust"
    assert data["display_name"] == "v3-robust (M12)"
    assert "v3-robust" in data["available_engines"]
    assert "efficientnet" in data["backbone"].lower()
    assert "transformer" in data["temporal_module"].lower()
    assert "FaceForensics++ C23 + Celeb-DF v2" in data["training_dataset"]


def test_predict_calibrated_real(client):
    """Verify POST /api/v1/predict returns calibrated REAL verdict with full metadata."""
    mock_pred = MagicMock(spec=VideoPredictor)
    mock_result = PredictionResult(
        video_path="sample.mp4",
        verdict="Real",
        probability=0.08,
        raw_logit=-2.44,
        confidence=0.92,
        threshold=0.5,
        frames_analyzed=16,
        faces_detected=16,
        device="cpu",
        model_version="best.pth",
        latency_ms={"preprocessing_ms": 110.0, "mc_dropout_inference_ms": 45.0, "ood_ms": 1.2, "total_ms": 156.2},
        decision="REAL",
        rejection_code=None,
        rejection_reason=None,
        uncertainty={
            "mean_probability": 0.08,
            "epistemic_variance": 0.0001,
            "entropy": 0.40,
            "mc_samples": 10,
            "ood_distance": 1.02,
            "is_ood": False,
            "is_uncertain": False,
        },
        quality={
            "total_frames": 16,
            "detected_frames": 16,
            "fallback_frames": 0,
            "failed_frames": 0,
            "fallback_ratio": 0.0,
            "is_acceptable": True,
            "rejection_reason": None,
        },
        model_info={
            "name": "v3-robust",
            "version": "m12",
            "architecture": "efficientnet_b0 + transformer",
            "device": "cpu",
        },
        calibration_info={
            "version": "m12_validation",
            "dataset": "M12 validation",
            "thresholds": {"classification_threshold": 0.5, "max_uncertainty_variance": 0.04, "ood_distance_threshold": 3.0},
        },
        diagnostics={
            "ood_status": "Within validated range",
            "uncertainty_status": "Low epistemic uncertainty",
            "quality_status": "Acceptable face detection",
        },
    )
    mock_pred.predict_video.return_value = mock_result
    set_predictor(mock_pred)

    fake_video = io.BytesIO(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00")
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("real_sample.mp4", fake_video, "video/mp4")},
        data={"engine": "v3-robust", "calibrated": "true"},
    )

    set_predictor(None)
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "Real"
    assert data["decision"] == "REAL"
    assert data["probability"] == 0.08
    assert data["confidence"] == 0.92
    assert data["rejection_code"] is None
    assert data["uncertainty"]["is_uncertain"] is False
    assert data["diagnostics"]["ood_status"] == "Within validated range"
    assert data["model_info"]["name"] == "v3-robust"
    assert data["engine"] == "v3-robust"


def test_predict_calibrated_compound_rejection(client):
    """Verify POST /api/v1/predict returns compound REJECT state without conflating with FAKE."""
    mock_pred = MagicMock(spec=VideoPredictor)
    mock_result = PredictionResult(
        video_path="noisy_sample.mp4",
        verdict="Reject",
        probability=0.55,
        raw_logit=0.20,
        confidence=0.0,
        threshold=0.5,
        frames_analyzed=16,
        faces_detected=3,
        device="cpu",
        model_version="best.pth",
        latency_ms={"preprocessing_ms": 95.0, "mc_dropout_inference_ms": 50.0, "ood_ms": 1.5, "total_ms": 146.5},
        decision="REJECT",
        rejection_code="LOW_QUALITY_AND_HIGH_UNCERTAINTY",
        rejection_reason="Low Video Quality: Insufficient face detections (3/16 < 8); High Epistemic Uncertainty: Model variance 0.0650 exceeds safety threshold 0.0400",
        uncertainty={
            "mean_probability": 0.55,
            "epistemic_variance": 0.065,
            "entropy": 0.99,
            "mc_samples": 10,
            "ood_distance": 3.85,
            "is_ood": True,
            "is_uncertain": True,
        },
        quality={
            "total_frames": 16,
            "detected_frames": 3,
            "fallback_frames": 13,
            "failed_frames": 0,
            "fallback_ratio": 0.8125,
            "is_acceptable": False,
            "rejection_reason": "Insufficient face detections (3/16 < 8); Excessive fallback crops (81.2% > 50.0%)",
        },
        model_info={"name": "v3-robust", "version": "m12", "architecture": "efficientnet_b0 + transformer", "device": "cpu"},
        calibration_info={"version": "m12_validation", "dataset": "M12 validation"},
        diagnostics={
            "ood_status": "Elevated feature distance (3.85 > 3.0)",
            "uncertainty_status": "Elevated variance (0.0650 > 0.04)",
            "quality_status": "Quality issue: Insufficient face detections",
        },
    )
    mock_pred.predict_video.return_value = mock_result
    set_predictor(mock_pred)

    fake_video = io.BytesIO(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00")
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("noisy_sample.mp4", fake_video, "video/mp4")},
        data={"engine": "v3-robust", "calibrated": "true"},
    )

    set_predictor(None)
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "Reject"
    assert data["decision"] == "REJECT"
    # Strict Semantic Rule: REJECT != FAKE
    assert data["decision"] != "FAKE"
    assert data["verdict"] != "Deepfake"
    assert data["rejection_code"] == "LOW_QUALITY_AND_HIGH_UNCERTAINTY"
    assert "Low Video Quality" in data["rejection_reason"]
    assert "High Epistemic Uncertainty" in data["rejection_reason"]
    assert data["confidence"] == 0.0
