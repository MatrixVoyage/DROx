"""Gate 4 Physical API Verification Script.

Tests:
1. GET /health
2. GET /api/v1/model-info
3. POST /api/v1/predict with real physical video Datasets/FaceForensics++_C23/original/004.mp4
Outputs: reports/gate4_api_audit.json
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from src.api.app import app, set_predictor
from src.inference.predictor import VideoPredictor


def main():
    print("=== Gate 4 API Physical Verification ===")
    video_path = Path("Datasets/FaceForensics++_C23/original/004.mp4")
    ckpt_path = Path("checkpoints/ffpp_c23_baseline_soft_lstm/best.pth")

    assert video_path.is_file(), f"Video not found: {video_path}"
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"

    # Initialize live VideoPredictor on GPU (or CPU if CUDA unavailable)
    predictor = VideoPredictor(checkpoint_path=ckpt_path)
    set_predictor(predictor)

    client = TestClient(app)

    # 1. Test /health
    health_resp = client.get("/health")
    print(f"GET /health: status={health_resp.status_code}")
    health_data = health_resp.json()
    assert health_resp.status_code == 200
    assert health_data["status"] == "healthy"
    assert health_data["model_loaded"] is True

    # 2. Test /api/v1/model-info
    info_resp = client.get("/api/v1/model-info")
    print(f"GET /api/v1/model-info: status={info_resp.status_code}")
    info_data = info_resp.json()
    assert info_resp.status_code == 200
    assert "ResNeXt-50" in info_data["architecture"]
    assert len(info_data["research_limitations"]) > 0

    # 3. Test POST /api/v1/predict with real physical video
    print(f"Uploading real video '{video_path}' to POST /api/v1/predict...")
    with open(video_path, "rb") as f:
        predict_resp = client.post(
            "/api/v1/predict",
            files={"file": (video_path.name, f, "video/mp4")},
            data={"threshold": 0.5},
        )
    print(f"POST /api/v1/predict: status={predict_resp.status_code}")
    assert predict_resp.status_code == 200
    predict_data = predict_resp.json()

    print(f"Prediction result:")
    print(f"  Verdict:         {predict_data.get('verdict')}")
    print(f"  Probability:     {predict_data.get('probability')}")
    print(f"  Raw Logit:       {predict_data.get('raw_logit')}")
    print(f"  Faces Detected:  {predict_data.get('faces_detected')}/16")
    print(f"  Latencies (ms):  {predict_data.get('latency_ms')}")

    audit_result = {
        "gate": "Gate 4 — API Service",
        "status": "PASS",
        "timestamp": "2026-09-21T02:05:00+05:30",
        "endpoints_verified": {
            "/health": health_data,
            "/api/v1/model-info": info_data,
            "/api/v1/predict": {
                "status_code": predict_resp.status_code,
                "input_video": str(video_path),
                "prediction": predict_data,
            },
        },
    }

    out_p = Path("reports/gate4_api_audit.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)
    print(f"Saved Gate 4 API audit to: {out_p}")


if __name__ == "__main__":
    main()
