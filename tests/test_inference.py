"""Unit and physical integration tests for VideoPredictor (Gate 1)."""

from pathlib import Path
import pytest
import torch

from src.inference.predictor import PredictionResult, VideoPredictor
from src.models.detector import DeepfakeDetector
from src.utils.config import load_config


CHECKPOINT_PATH = Path("checkpoints/ffpp_c23_baseline_soft_lstm/best.pth")
REAL_VIDEO_PATH = Path("Datasets/FaceForensics++_C23/original/004.mp4")
FAKE_VIDEO_PATH = Path("Datasets/FaceForensics++_C23/Deepfakes/004_982.mp4")


def test_prediction_result_serialization():
    """Verify PredictionResult fields and dictionary conversion."""
    res = PredictionResult(
        video_path="test.mp4",
        verdict="Deepfake",
        probability=0.88,
        raw_logit=2.01,
        confidence=0.88,
        threshold=0.5,
        frames_analyzed=16,
        faces_detected=16,
        device="cpu",
        model_version="best.pth",
        latency_ms={"model_forward_ms": 45.2, "total_ms": 120.5},
        error=None,
    )
    d = res.to_dict()
    assert d["verdict"] == "Deepfake"
    assert d["probability"] == 0.88
    assert d["raw_logit"] == 2.01
    assert d["confidence"] == 0.88
    assert d["frames_analyzed"] == 16


def test_predict_tensor_contract(tmp_path):
    """Verify input shape validation and deterministic evaluation on synthetic tensor."""
    config = load_config("configs/default.yaml")
    model = DeepfakeDetector(config.model)

    # Save mock checkpoint
    mock_ckpt = tmp_path / "mock.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict()}, mock_ckpt)

    predictor = VideoPredictor(mock_ckpt, device="cpu")

    # Valid 4D tensor (T, 3, 112, 112)
    x4d = torch.randn(16, 3, 112, 112, dtype=torch.float32)
    res1 = predictor.predict_tensor(x4d)
    assert res1.verdict in ("Real", "Deepfake")
    assert 0.0 <= res1.probability <= 1.0
    assert 0.5 <= res1.confidence <= 1.0
    assert res1.frames_analyzed == 16

    # Determinism test on identical input
    res2 = predictor.predict_tensor(x4d)
    assert abs(res1.raw_logit - res2.raw_logit) < 1e-6
    assert abs(res1.probability - res2.probability) < 1e-6

    # Invalid tensor dimensions
    with pytest.raises(ValueError, match="Expected 5D tensor"):
        predictor.predict_tensor(torch.randn(3, 112, 112))

    # Invalid spatial dimensions
    with pytest.raises(ValueError, match="spatial dims"):
        predictor.predict_tensor(torch.randn(1, 16, 3, 64, 64))

    # Non-finite values
    bad_tensor = x4d.clone()
    bad_tensor[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        predictor.predict_tensor(bad_tensor)


def test_predict_file_not_found(tmp_path):
    """Verify graceful handling when video path does not exist."""
    config = load_config("configs/default.yaml")
    model = DeepfakeDetector(config.model)
    mock_ckpt = tmp_path / "mock.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict()}, mock_ckpt)

    predictor = VideoPredictor(mock_ckpt, device="cpu")
    res = predictor.predict_video("non_existent_file.mp4")
    assert res.verdict == "Error"
    assert "FILE_NOT_FOUND" in res.error


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="best.pth not found")
def test_gate1_physical_videos_evaluation():
    """Gate 1 physical verification: Evaluate both real and fake physical FF++ videos using best.pth."""
    assert REAL_VIDEO_PATH.exists(), f"Missing real video: {REAL_VIDEO_PATH}"
    assert FAKE_VIDEO_PATH.exists(), f"Missing fake video: {FAKE_VIDEO_PATH}"

    # Use CUDA if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = VideoPredictor(CHECKPOINT_PATH, device=device)

    # 1. Evaluate REAL physical video
    real_res = predictor.predict_video(REAL_VIDEO_PATH)
    assert real_res.error is None, f"Real video prediction failed: {real_res.error}"
    assert real_res.frames_analyzed == 16
    assert real_res.faces_detected > 0
    assert 0.0 <= real_res.probability <= 1.0

    # 2. Evaluate FAKE physical video
    fake_res = predictor.predict_video(FAKE_VIDEO_PATH)
    assert fake_res.error is None, f"Fake video prediction failed: {fake_res.error}"
    assert fake_res.frames_analyzed == 16
    assert fake_res.faces_detected > 0
    assert 0.0 <= fake_res.probability <= 1.0

    # 3. Determinism check: second pass on fake video
    fake_res_repeat = predictor.predict_video(FAKE_VIDEO_PATH)
    assert abs(fake_res.raw_logit - fake_res_repeat.raw_logit) < 1e-4
    assert abs(fake_res.probability - fake_res_repeat.probability) < 1e-4

import tempfile
import shutil
from fastapi.testclient import TestClient
from src.api.app import app

@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="best.pth not found")
def test_inference_parity_with_video_id():
    """M11 Inference Parity and Determinism Tests."""
    assert REAL_VIDEO_PATH.exists(), f"Missing real video: {REAL_VIDEO_PATH}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = VideoPredictor(CHECKPOINT_PATH, device=device)

    # a. Explicit video_id produces deterministic sampling
    vid = "stable_id_123"
    res1 = predictor.predict_video(REAL_VIDEO_PATH, video_id=vid)
    res2 = predictor.predict_video(REAL_VIDEO_PATH, video_id=vid)
    assert res1.metadata["frame_indices"] == res2.metadata["frame_indices"], "Deterministic sampling failed"

    # b. Same video + different temporary filename + same video_id produces identical frame indices
    # c. Same video + same video_id produces identical score
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path_1 = Path(tmp.name)
        shutil.copy(REAL_VIDEO_PATH, tmp_path_1)
    
    try:
        res_tmp = predictor.predict_video(tmp_path_1, video_id=vid)
        assert res1.metadata["frame_indices"] == res_tmp.metadata["frame_indices"], "Different filename broke parity"
        assert abs(res1.probability - res_tmp.probability) < 1e-5, "Different filename broke score parity"
    finally:
        tmp_path_1.unlink()

    # d. Different video_id is allowed to produce different deterministic sampling
    res_diff_id = predictor.predict_video(REAL_VIDEO_PATH, video_id="different_id_456")
    # Highly likely to be different due to RNG seed
    assert res1.metadata["frame_indices"] != res_diff_id.metadata["frame_indices"], "Different ID should yield different frames"

    # f. Existing inference behavior remains backward compatible when video_id is omitted
    res_omitted = predictor.predict_video(REAL_VIDEO_PATH)
    assert res_omitted.metadata["frame_indices"] is not None
    assert isinstance(res_omitted.probability, float)

def test_api_upload_identity_derivation():
    """e. API upload does not derive sampling identity from the temporary filename."""
    client = TestClient(app)
    # Using the existing REAL_VIDEO_PATH to simulate an upload
    assert REAL_VIDEO_PATH.exists(), f"Missing real video: {REAL_VIDEO_PATH}"
    
    with open(REAL_VIDEO_PATH, "rb") as f:
        # We explicitly supply a filename 'orig_004.mp4'
        response = client.post(
            "/api/v1/predict",
            files={"file": ("orig_004.mp4", f, "video/mp4")},
            data={"engine": "v1-baseline"}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["original_filename"] == "orig_004.mp4"
    # Verify metadata shows the correct video_id derived from filename stem ('orig_004')
    if "metadata" in data and "video_id" in data["metadata"]:
        assert data["metadata"]["video_id"] == "orig_004"

