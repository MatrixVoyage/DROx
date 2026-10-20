import sys
from pathlib import Path
import json
import shutil
import tempfile

import pandas as pd

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.inference.predictor import VideoPredictor

def run_parity_audit():
    print("Loading manifest and selecting 5 real + 5 fake test videos...")
    manifest_df = pd.read_csv("manifests/ffpp_c23.csv")
    test_df = manifest_df[manifest_df["split"] == "test"]
    real_videos = test_df[test_df["label"] == 0].head(5)
    fake_videos = test_df[test_df["label"] == 1].head(5)
    subset_df = pd.concat([real_videos, fake_videos])

    print("Initializing predictor...")
    predictor = VideoPredictor(
        checkpoint_path="checkpoints/m10_selected_candidate/best.pth",
        threshold=0.5
    )

    out_dir = Path("reports/m11_inference_parity")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for idx, row in subset_df.iterrows():
        vpath = Path(row["video_path"])
        vid = str(row["video_id"])
        
        print(f"Auditing video: {vid}")
        
        # A. Benchmark/control path
        res_ctrl = predictor.predict_video(str(vpath), video_id=vid)
        
        # B. API-simulation path
        with tempfile.NamedTemporaryFile(suffix=vpath.suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            shutil.copy(vpath, tmp_path)
        
        try:
            res_api = predictor.predict_video(str(tmp_path), video_id=vid)
        finally:
            tmp_path.unlink()
            
        score_diff = abs(res_ctrl.probability - res_api.probability)
        
        # extract frame indices from metadata
        meta_ctrl = res_ctrl.metadata or {}
        meta_api = res_api.metadata or {}
        
        indices_ctrl = meta_ctrl.get("frame_indices", [])
        indices_api = meta_api.get("frame_indices", [])
        
        results.append({
            "video_id": vid,
            "original_path": str(vpath),
            "temporary_path": str(tmp_path),
            "exact_sampled_frame_indices_control": indices_ctrl,
            "exact_sampled_frame_indices_api": indices_api,
            "preprocessing_metadata_control": meta_ctrl,
            "preprocessing_metadata_api": meta_api,
            "model_score_control": res_ctrl.probability,
            "model_score_api": res_api.probability,
            "predicted_label_control": res_ctrl.verdict,
            "predicted_label_api": res_api.verdict,
            "absolute_score_difference": score_diff,
            "parity_achieved": score_diff < 1e-5
        })
        
    with open(out_dir / "parity_audit.json", "w") as f:
        json.dump({"parity_results": results}, f, indent=2)
        
    print(f"Audit complete! Saved to {out_dir / 'parity_audit.json'}")

if __name__ == "__main__":
    run_parity_audit()
