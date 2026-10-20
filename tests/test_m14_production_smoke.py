"""Milestone 14 Phase A0: Production Smoke & Application Freeze Gate.

Verifies the end-to-end production application stack before any M14 research experiments begin:
1. Frozen Checkpoint SHA-256 hashes (M12, M13.2b, v5 champion)
2. Calibration config integrity (T=2.0502, score_space=temperature_scaled_probability, tau*=0.1260)
3. VideoPredictor dynamic instantiation and contract
4. FastAPI service endpoints (/health, /api/v1/model-info, /api/v1/predict, /)
5. Physical video integration smoke tests (Real 004.mp4 and Fake 004_982.mp4)
"""

import hashlib
import json
from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient

from src.api.app import app, get_predictor
from src.inference.predictor import VideoPredictor
from src.models.family_invariant_detector import FamilyInvariantDeepfakeDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
M12_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "best.pth"
M13_2B_PATH = PROJECT_ROOT / "checkpoints" / "m13_2b_stage5" / "best.pth"
M13_4_PATH = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "best.pth"
CALIB_PATH = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "calibration_config.json"

M12_EXPECTED_HASH = "c8265f634880c56929e5d82be8e4fc808fc7776d0750fca86c74e18a983cda56"
M13_2B_EXPECTED_HASH = "d681386102f1d562b15ee9b7fb68eef7e4e08c1c41a8c22db6e1654c11e67611"
V5_CHAMPION_EXPECTED_HASH = "4e81b49308bbf14993c3de43377afd3349945f6e5a69ed4af307f4f26009dfc7"

REAL_VIDEO_PATH = PROJECT_ROOT / "Datasets" / "FaceForensics++_C23" / "original" / "004.mp4"
FAKE_VIDEO_PATH = PROJECT_ROOT / "Datasets" / "FaceForensics++_C23" / "Deepfakes" / "004_982.mp4"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_phase_a0_frozen_checkpoint_hashes():
    """Verify bitwise immutability of M12 baseline, M13.2b reference, and v5 champion."""
    assert M12_PATH.is_file(), f"Missing M12 checkpoint at {M12_PATH}"
    assert sha256_file(M12_PATH) == M12_EXPECTED_HASH, "M12 hash mismatch"

    assert M13_2B_PATH.is_file(), f"Missing M13.2b checkpoint at {M13_2B_PATH}"
    assert sha256_file(M13_2B_PATH) == M13_2B_EXPECTED_HASH, "M13.2b hash mismatch"

    assert M13_4_PATH.is_file(), f"Missing v5 champion checkpoint at {M13_4_PATH}"
    assert sha256_file(M13_4_PATH) == V5_CHAMPION_EXPECTED_HASH, "v5 champion hash mismatch"


def test_phase_a0_calibration_config_integrity():
    """Verify calibration_config.json specifies temperature_scaled_probability and tau*=0.1260."""
    assert CALIB_PATH.is_file(), f"Missing calibration_config.json at {CALIB_PATH}"
    with open(CALIB_PATH, "r") as f:
        calib_data = json.load(f)

    assert calib_data["score_space"] == "temperature_scaled_probability"
    assert calib_data["temperature"] == 2.0502
    assert calib_data["threshold_standard"] == 0.5000
    assert calib_data["threshold_high_sensitivity"] == 0.1260
    assert calib_data["threshold_selection_source"] == "validation_only"

    # Production decision pipeline block
    assert "production_decision_pipeline" in calib_data
    pipe = calib_data["production_decision_pipeline"]
    assert pipe["score_space"] == "temperature_scaled_probability"
    assert pipe["temperature"] == 2.0502
    assert pipe["operating_thresholds"]["high_sensitivity_spec90"] == 0.1260

    # Diagnostic OOD contract
    assert calib_data["uncertainty_safety"]["ood_status"] == "diagnostic"
    assert calib_data["uncertainty_safety"]["reject_on_ood"] is False


def test_phase_a0_predictor_instantiation_and_contract():
    """Verify VideoPredictor dynamically loads v5-family-invariant on CPU/CUDA cleanly."""
    predictor = VideoPredictor(checkpoint_path=M13_4_PATH, device="cpu")
    assert isinstance(predictor.model, FamilyInvariantDeepfakeDetector)
    info = predictor._get_model_info_dict()
    assert info["name"] == "v5-family-invariant"
    assert info["version"] == "m13.4"
    assert "LocalFreq" in info["architecture"]
    assert "LocalTexture" in info["architecture"]


def test_phase_a0_fastapi_endpoints():
    """Verify FastAPI endpoints (/health, /api/v1/model-info, /) return 200 with complete data."""
    client = TestClient(app)

    # Health check
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    health_data = resp_health.json()
    assert health_data["status"] == "healthy"
    assert "v5-family-invariant" in health_data["available_engines"]

    # Model info
    resp_info = client.get("/api/v1/model-info?engine=v5-family-invariant")
    assert resp_info.status_code == 200
    info_data = resp_info.json()
    assert info_data["engine"] == "v5-family-invariant"
    assert "checkpoint_provenance" in info_data
    prov = info_data["checkpoint_provenance"]
    assert prov["single_deployed_checkpoint"]["held_out_f2f_test_auc"] == 0.7903
    assert prov["research_family_transfer_study"]["macro_mean_auc"] == 0.9165

    # UI Dashboard static mount
    resp_ui = client.get("/")
    assert resp_ui.status_code == 200
    assert "Deepfake Video Detection" in resp_ui.text


def test_phase_a0_physical_video_smoke_tests():
    """Verify physical real and fake videos smoke test cleanly through the production predictor."""
    if not REAL_VIDEO_PATH.is_file() or not FAKE_VIDEO_PATH.is_file():
        pytest.skip("Physical test videos missing")

    predictor = get_predictor("v5-family-invariant")

    # Real Video
    res_real = predictor.predict_video(REAL_VIDEO_PATH, calibrated=True)
    assert res_real.verdict == "Real"
    assert res_real.decision == "REAL"
    assert res_real.probability < 0.10

    # Fake Video
    res_fake = predictor.predict_video(FAKE_VIDEO_PATH, calibrated=True)
    assert res_fake.verdict == "Deepfake"
    assert res_fake.decision == "FAKE"
    assert res_fake.probability > 0.90
