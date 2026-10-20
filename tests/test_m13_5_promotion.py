"""Unit tests for Milestone 13.5: Consolidation, Promotion & Explicit Limitations."""

import hashlib
import json
from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient

from src.api.app import app
from src.inference.predictor import VideoPredictor
from src.models.family_invariant_detector import FamilyInvariantDeepfakeDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
M12_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "best.pth"
M13_2B_PATH = PROJECT_ROOT / "checkpoints" / "m13_2b_stage5" / "best.pth"
M13_4_PATH = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "best.pth"

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


def test_frozen_checkpoint_hashes_immutable():
    """Verify bitwise immutability of M12 baseline, M13.2b reference, and frozen v5 champion."""
    assert M12_PATH.is_file(), f"M12 checkpoint missing at {M12_PATH}"
    assert sha256_file(M12_PATH) == M12_EXPECTED_HASH, "M12 hash mismatch"

    assert M13_2B_PATH.is_file(), f"M13.2b checkpoint missing at {M13_2B_PATH}"
    assert sha256_file(M13_2B_PATH) == M13_2B_EXPECTED_HASH, "M13.2b hash mismatch"

    assert M13_4_PATH.is_file(), f"v5 champion checkpoint missing at {M13_4_PATH}"
    assert sha256_file(M13_4_PATH) == V5_CHAMPION_EXPECTED_HASH, "v5 champion hash mismatch"


def test_v5_champion_predictor_instantiation():
    """Verify VideoPredictor automatically detects and instantiates FamilyInvariantDeepfakeDetector."""
    assert M13_4_PATH.is_file(), f"M13.4 champion checkpoint missing at {M13_4_PATH}"
    predictor = VideoPredictor(checkpoint_path=M13_4_PATH, device="cpu")
    assert isinstance(predictor.model, FamilyInvariantDeepfakeDetector)
    info = predictor._get_model_info_dict()
    assert info["name"] == "v5-family-invariant"
    assert info["version"] == "m13.4"
    assert "LocalFreq" in info["architecture"]
    assert "LocalTexture" in info["architecture"]


def test_calibration_and_operating_thresholds_separated():
    """Verify calibration (temperature scaling) is strictly separated from operating threshold policies."""
    calib_cfg_path = M13_4_PATH.parent / "calibration_config.json"
    assert calib_cfg_path.is_file(), "calibration_config.json missing"
    with open(calib_cfg_path, "r") as f:
        calib_data = json.load(f)

    # Temperature scaling is for ECE calibration
    assert calib_data["temperature"] == 2.0502
    assert "temperature_scaling" in calib_data
    assert calib_data["temperature_scaling"]["temperature"] == 2.0502

    # Operating thresholds are decision boundaries, defined in the production score space
    assert "production_decision_pipeline" in calib_data
    assert calib_data["score_space"] == "temperature_scaled_probability"
    op_thresh = calib_data["operating_thresholds"]
    assert op_thresh["score_space"] == "temperature_scaled_probability"
    assert op_thresh["standard"] == 0.50
    assert op_thresh["high_sensitivity_spec90"] == 0.1260
    assert op_thresh["youden_optimal"] == 0.1912
    assert op_thresh["high_specificity_spec95"] == 0.6820

    # Raw score space reference preserved for research transparency
    assert "raw_score_space_reference" in calib_data
    raw_thresh = calib_data["raw_score_space_reference"]["operating_thresholds"]
    assert raw_thresh["high_sensitivity_spec90"] == 0.0190

    # OOD provenance and diagnostic-only status
    assert calib_data["ood_distance_threshold"] == 3.18
    assert "ood_threshold_provenance" in calib_data["uncertainty_safety"]
    assert "Diagnostic indicator only" in calib_data["uncertainty_safety"]["ood_status_note"]
    assert calib_data["uncertainty_safety"]["reject_on_ood"] is False


def test_branch_scores_are_diagnostic_probabilities_not_contributions():
    """Verify forward pass exposes branch scores as raw output probabilities, not attribution percentages."""
    predictor = VideoPredictor(checkpoint_path=M13_4_PATH, device="cpu")
    dummy = torch.randn(1, 16, 3, 112, 112, dtype=torch.float32)

    # Test uncalibrated pass
    res_uncal = predictor.predict_tensor(dummy, calibrated=False)
    assert res_uncal.metadata is not None
    assert "branch_scores" in res_uncal.metadata
    b_scores = res_uncal.metadata["branch_scores"]
    for key in ("p_fused", "p_composite", "p_freq", "p_texture", "p_aux_fake"):
        assert key in b_scores
        assert 0.0 <= b_scores[key] <= 1.0

    # Verify they are independent probabilities, NOT percentages summing to 100
    sum_scores = sum(b_scores[k] for k in ("p_composite", "p_freq", "p_texture", "p_aux_fake"))
    # They shouldn't sum to 1.0 or 100 because they are independent branch sigmoid outputs
    assert sum_scores != 1.0

    # Test calibrated pass exposes raw and calibrated probabilities plus branch scores
    res_cal = predictor.predict_tensor(dummy, calibrated=True)
    assert res_cal.metadata is not None
    assert "raw_probability" in res_cal.metadata
    assert "calibrated_probability" in res_cal.metadata
    assert "branch_scores" in res_cal.metadata
    assert "branch_scores" in res_cal.diagnostics


def test_predictor_and_api_parity_and_determinism():
    """Verify exact parity between VideoPredictor and FastAPI endpoint, plus repeated determinism."""
    if not REAL_VIDEO_PATH.is_file():
        pytest.skip(f"Test video missing at {REAL_VIDEO_PATH}")

    from src.api.app import get_predictor
    predictor = get_predictor("v5-family-invariant")

    # Run direct predictor twice to check determinism
    res1 = predictor.predict_video(REAL_VIDEO_PATH, calibrated=False)
    res2 = predictor.predict_video(REAL_VIDEO_PATH, calibrated=False)

    assert res1.probability == res2.probability, "Nondeterministic prediction probabilities"
    assert res1.raw_logit == res2.raw_logit, "Nondeterministic raw logits"
    assert res1.verdict == res2.verdict, "Nondeterministic verdicts"

    # Run through FastAPI endpoint
    client = TestClient(app)
    with open(REAL_VIDEO_PATH, "rb") as f:
        api_resp = client.post(
            "/api/v1/predict",
            files={"file": ("004.mp4", f, "video/mp4")},
            data={"engine": "v5-family-invariant", "calibrated": "false"},
        )
    assert api_resp.status_code == 200
    api_data = api_resp.json()

    # Exact Parity Checks
    assert api_data["verdict"] == res1.verdict
    assert api_data["probability"] == res1.probability
    assert abs(api_data["raw_logit"] - res1.raw_logit) < 1e-4
    assert api_data["engine"] == "v5-family-invariant"

    # Branch scores parity
    api_branches = api_data["metadata"]["branch_scores"]
    direct_branches = res1.metadata["branch_scores"]
    for k in ("p_fused", "p_composite", "p_freq", "p_texture", "p_aux_fake"):
        assert api_branches[k] == direct_branches[k]


def test_physical_video_e2e_real_and_fake():
    """Verify physical real and fake videos evaluate end-to-end correctly on v5 champion."""
    if not REAL_VIDEO_PATH.is_file() or not FAKE_VIDEO_PATH.is_file():
        pytest.skip("Physical test videos missing")

    from src.api.app import get_predictor
    predictor = get_predictor("v5-family-invariant")

    # Real video test
    res_real = predictor.predict_video(REAL_VIDEO_PATH, calibrated=True)
    assert res_real.verdict == "Real"
    assert res_real.decision == "REAL"
    assert res_real.probability < 0.10
    assert res_real.confidence > 0.90
    assert "Within validated range" in res_real.diagnostics["ood_status"]

    # Fake video test
    res_fake = predictor.predict_video(FAKE_VIDEO_PATH, calibrated=True)
    assert res_fake.verdict == "Deepfake"
    assert res_fake.decision == "FAKE"
    assert res_fake.probability > 0.90
    assert res_fake.confidence > 0.90


def test_api_model_info_v5_explicit_limitations_and_provenance():
    """Verify FastAPI /api/v1/model-info discloses explicit limitations and separates LOFO-F2F single checkpoint from matrix aggregate."""
    client = TestClient(app)
    response = client.get("/api/v1/model-info?engine=v5-family-invariant")
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == "v5-family-invariant"
    assert "v5-family-invariant" in data["available_engines"]

    # Checkpoint Provenance Separation Gate
    assert "checkpoint_provenance" in data
    prov = data["checkpoint_provenance"]
    assert prov["training_fold"] == "LOFO-F2F (seen: Deepfakes, FaceSwap, NeuralTextures; held-out: Face2Face)"
    assert prov["held_out_f2f_test_auc"] == 0.7903
    assert prov["lofo_matrix_macro_mean_auc"] == 0.9165
    assert "aggregate macro-mean across 4 distinct models" in prov["note"]

    # Also assert structured separation
    single_ckpt = prov["single_deployed_checkpoint"]
    assert single_ckpt["training_fold"] == "LOFO-F2F (seen: Deepfakes, FaceSwap, NeuralTextures; held-out: Face2Face)"
    assert single_ckpt["held_out_f2f_test_auc"] == 0.7903
    study = prov["research_family_transfer_study"]
    assert study["macro_mean_auc"] == 0.9165
    assert "aggregate research-matrix statistic across 4 distinct models" in study["note"]

    # 5 Explicit Research Limitations Gate + OOD Diagnostic Status
    assert "research_limitations" in data
    limits = " ".join(data["research_limitations"])
    assert "Facial Reenactment Manifold Boundary" in limits
    assert "Handcrafted Branch Gating" in limits
    assert "Threshold Sensitivity Trade-Off" in limits
    assert "Cross-Dataset Generalization" in limits
    assert "Probabilistic Confidence Semantics" in limits
    assert "OOD Status" in limits
