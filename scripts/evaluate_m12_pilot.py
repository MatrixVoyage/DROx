"""M12 Pilot Evaluation Script.

Evaluates the M12 pilot checkpoint against the frozen M10 baseline
on the 404-video held-out pilot test split.

Outputs comparison metrics (AUC, Acc, EER) for both models to demonstrate
whether the pilot improvements are plausible before scaling up.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import VideoPredictor
from src.evaluation.metrics import compute_eer
from src.utils.logger import setup_logger

logger = setup_logger("m12_pilot_eval")

MANIFEST_PATH = PROJECT_ROOT / "manifests" / "mixed_robust_train_p20_seed42.csv"
M10_CKPT = PROJECT_ROOT / "checkpoints" / "m10_selected_candidate" / "best.pth"
M12_CKPT = PROJECT_ROOT / "checkpoints" / "m12_pilot" / "best.pth"
REPORT_PATH = PROJECT_ROOT / "reports" / "m12_pilot_comparison.json"


def evaluate_model(
    predictor: VideoPredictor,
    df: pd.DataFrame,
    model_name: str,
) -> dict:
    """Evaluate a predictor on the test dataframe."""
    logger.info(f"Evaluating {model_name} on {len(df)} videos...")

    y_true = []
    y_prob = []
    y_pred = []
    
    t0 = time.time()

    for idx, row in df.iterrows():
        vid_path = PROJECT_ROOT / row["video_path"]
        
        # We catch any potential inference errors
        try:
            res = predictor.predict_video(vid_path, video_id=row["video_id"])
            if res.error:
                logger.warning(f"Error on {row['video_id']}: {res.error}")
                continue
            
            y_true.append(row["label"])
            y_prob.append(res.probability)
            y_pred.append(1 if res.verdict == "Deepfake" else 0)
        except Exception as e:
            logger.error(f"Failed {row['video_id']}: {e}")
            continue
            
        if (idx + 1) % 50 == 0:
            logger.info(f"  Processed {idx + 1}/{len(df)}...")

    duration = time.time() - t0

    # Compute metrics
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    eer = compute_eer(y_true, y_prob)

    metrics = {
        "model": model_name,
        "videos_evaluated": len(y_true),
        "duration_seconds": round(duration, 2),
        "roc_auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "eer": round(eer, 4)
    }
    
    logger.info(f"[{model_name}] AUC: {auc:.4f} | Acc: {acc:.4f} | EER: {eer:.4f}")
    return metrics


def main() -> None:
    print("=" * 70)
    print("M12.2 Pilot Evaluation vs M10 Baseline")
    print("=" * 70)

    # 1. Load manifest and filter to test split
    df = pd.read_csv(MANIFEST_PATH)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    logger.info(f"Loaded {len(test_df)} test videos from pilot manifest.")

    if not M10_CKPT.exists():
        raise FileNotFoundError(f"M10 baseline checkpoint not found: {M10_CKPT}")
    if not M12_CKPT.exists():
        raise FileNotFoundError(f"M12 pilot checkpoint not found: {M12_CKPT}")

    # 2. Initialize predictors
    logger.info("Initializing M10 Predictor...")
    predictor_m10 = VideoPredictor(checkpoint_path=M10_CKPT, device="cuda")
    
    logger.info("Initializing M12 Predictor...")
    predictor_m12 = VideoPredictor(checkpoint_path=M12_CKPT, device="cuda")

    # 3. Evaluate both
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    m10_metrics = evaluate_model(predictor_m10, test_df, "M10_Baseline")
    m12_metrics = evaluate_model(predictor_m12, test_df, "M12_Pilot")

    # 4. Save results
    report = {
        "m10_baseline": m10_metrics,
        "m12_pilot": m12_metrics,
        "improvement_auc": round(m12_metrics["roc_auc"] - m10_metrics["roc_auc"], 4),
        "improvement_acc": round(m12_metrics["accuracy"] - m10_metrics["accuracy"], 4),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Comparison report saved to {REPORT_PATH}")
    logger.info(f"AUC Improvement: {report['improvement_auc']:+0.4f}")


if __name__ == "__main__":
    main()
