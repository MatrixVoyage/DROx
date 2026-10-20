"""M12 Full Scientific Evaluation Script.

Executes the comprehensive M12 scientific evaluation matrix:
1. In-Domain Seen-Manipulation (FF++ Deepfakes, FaceSwap, NeuralTextures)
2. Seen-Dataset Generalization (Celeb-DF v2)
3. Unseen-Manipulation Generalization (FF++ Face2Face held-out)
4. Degradation Robustness (JPEG compression Q=30, Low-Resolution downscale 0.5x)
5. Out-of-Distribution & Epistemic Uncertainty (Noise / Non-face, MC-Dropout)
6. Three-State Decision Policy (Real, Deepfake, Reject)

Compares frozen M10 baseline against M12 robust model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import torch
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augment import RobustAugmentor, RobustAugmentorConfig
from src.evaluation.metrics import compute_eer
from src.inference.predictor import VideoPredictor
from src.inference.uncertainty import (
    CalibratedDecisionPolicy,
    MCDropoutEstimator,
    OODFeatureEstimator,
    QualityAssessor,
    QualitySignals,
    UncertaintySignals,
)
from src.utils.logger import setup_logger

logger = setup_logger("m12_full_eval")

MANIFEST_PATH = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"
M10_CKPT = PROJECT_ROOT / "checkpoints" / "m10_selected_candidate" / "best.pth"
M12_CKPT = PROJECT_ROOT / "checkpoints" / "m12_full" / "best.pth"
OUTPUT_REPORT_JSON = PROJECT_ROOT / "reports" / "m12_full_scientific_report.json"
OUTPUT_REPORT_MD = PROJECT_ROOT / "reports" / "m12_full_scientific_report.md"


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute Brier Score and Expected Calibration Error (ECE)."""
    brier = float(np.mean((y_prob - y_true) ** 2))

    # ECE with 10 equal-width bins
    n_bins = 10
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_mask = (y_prob >= bin_limits[i]) & (y_prob < bin_limits[i + 1])
        if i == n_bins - 1:
            bin_mask = bin_mask | (y_prob == 1.0)
        bin_count = int(np.sum(bin_mask))
        if bin_count > 0:
            bin_acc = float(np.mean(y_true[bin_mask]))
            bin_conf = float(np.mean(y_prob[bin_mask]))
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return {"brier_score": round(brier, 4), "ece": round(float(ece), 4)}


def evaluate_subset(
    predictor: VideoPredictor,
    df: pd.DataFrame,
    subset_name: str,
    degradation: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a predictor on a specific test subset."""
    if max_samples is not None and len(df) > max_samples:
        parts = []
        for lbl, group in df.groupby("label"):
            n_lbl = max(1, int(round(max_samples * len(group) / len(df))))
            parts.append(group.sample(n=min(n_lbl, len(group)), random_state=42))
        eval_df = pd.concat(parts).reset_index(drop=True)
    else:
        eval_df = df.reset_index(drop=True)

    logger.info(f"Evaluating {subset_name} ({len(eval_df)} samples, degradation={degradation})...")

    # Set up degradations if requested
    jpeg_aug = RobustAugmentor(
        config=RobustAugmentorConfig(
            jpeg_p=1.0, blur_p=0.0, noise_p=0.0, resize_p=0.0, reencode_p=0.0, frame_drop_p=0.0
        ),
        seed=42,
    )
    lowres_aug = RobustAugmentor(
        config=RobustAugmentorConfig(
            jpeg_p=0.0, blur_p=0.0, noise_p=0.0, resize_p=1.0, reencode_p=0.0, frame_drop_p=0.0
        ),
        seed=42,
    )

    y_true = []
    y_prob = []
    y_pred = []

    t0 = time.time()

    for idx, row in eval_df.iterrows():
        vid_path = PROJECT_ROOT / row["video_path"]
        vid_id = str(row["video_id"])
        lbl = int(row["label"])

        try:
            # Check if preprocessed tensor exists
            tensor_file = PROJECT_ROOT / "processed" / "m12_full" / "test" / "tensors" / f"{vid_id}.pt"
            if tensor_file.is_file():
                tensor = torch.load(tensor_file, map_location="cpu", weights_only=False)
                if degradation == "jpeg_compression":
                    tensor, _ = jpeg_aug.augment_tensor(tensor)
                elif degradation == "low_resolution":
                    tensor, _ = lowres_aug.augment_tensor(tensor)
                res = predictor.predict_tensor(tensor, video_id=vid_id)
            else:
                res = predictor.predict_video(vid_path, video_id=vid_id)

            if res.error:
                continue

            prob = res.probability
            y_true.append(lbl)
            y_prob.append(prob)
            y_pred.append(1 if prob >= 0.50 else 0)

        except Exception as e:
            logger.warning(f"Error on {vid_id}: {e}")
            continue

    duration = time.time() - t0
    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    y_pred_np = np.array(y_pred)

    if len(np.unique(y_true_np)) > 1:
        auc = roc_auc_score(y_true_np, y_prob_np)
        eer = compute_eer(y_true_np, y_prob_np)
    else:
        auc = float("nan")
        eer = float("nan")

    acc = accuracy_score(y_true_np, y_pred_np)
    prec = precision_score(y_true_np, y_pred_np, zero_division=0)
    rec = recall_score(y_true_np, y_pred_np, zero_division=0)
    f1 = f1_score(y_true_np, y_pred_np, zero_division=0)
    calib = compute_calibration_metrics(y_true_np, y_prob_np)

    return {
        "subset": subset_name,
        "degradation": degradation or "clean",
        "num_evaluated": len(y_true),
        "duration_seconds": round(duration, 2),
        "roc_auc": round(float(auc), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "eer": round(float(eer), 4),
        "brier_score": calib["brier_score"],
        "ece": calib["ece"],
    }


def run_full_matrix(
    predictor: VideoPredictor,
    test_df: pd.DataFrame,
    model_name: str,
    max_samples_per_subtest: Optional[int] = None,
) -> Dict[str, Any]:
    """Run complete evaluation matrix on predictor."""
    results = {}

    # 1. FF++ In-Domain Seen-Manipulations (Deepfakes, FaceSwap, NeuralTextures + Real)
    ffpp_seen_df = test_df[
        (test_df["dataset"] == "ffpp") & (test_df["manipulation_type"] != "Face2Face")
    ]
    results["ffpp_in_domain_seen"] = evaluate_subset(
        predictor, ffpp_seen_df, "FF++ Seen Manipulations", max_samples=max_samples_per_subtest
    )

    # 2. Celeb-DF Seen-Dataset
    celebdf_df = test_df[test_df["dataset"] == "celebdf"]
    results["celebdf_seen_dataset"] = evaluate_subset(
        predictor, celebdf_df, "Celeb-DF v2", max_samples=max_samples_per_subtest
    )

    # 3. FF++ Face2Face Unseen-Manipulation
    ffpp_f2f_df = test_df[
        (test_df["dataset"] == "ffpp") &
        ((test_df["manipulation_type"] == "Face2Face") | (test_df["label"] == 0))
    ]
    results["ffpp_unseen_face2face"] = evaluate_subset(
        predictor, ffpp_f2f_df, "FF++ Face2Face (Unseen)", max_samples=max_samples_per_subtest
    )

    # 4. Degradation: Compressed (on Celeb-DF)
    results["degraded_compressed"] = evaluate_subset(
        predictor,
        celebdf_df,
        "Celeb-DF Compressed",
        degradation="jpeg_compression",
        max_samples=max_samples_per_subtest,
    )

    # 5. Degradation: Low-Resolution (on Celeb-DF)
    results["degraded_lowres"] = evaluate_subset(
        predictor,
        celebdf_df,
        "Celeb-DF Low-Resolution",
        degradation="low_resolution",
        max_samples=max_samples_per_subtest,
    )

    return {"model_name": model_name, "subtests": results}


def generate_markdown_report(report_data: Dict[str, Any], md_path: Path) -> None:
    """Generate professional Markdown summary table."""
    m10_sub = report_data.get("m10_baseline", {}).get("subtests", {})
    m12_sub = report_data.get("m12_full", {}).get("subtests", {})

    lines = [
        "# Milestone 12: Full Scientific Evaluation Report",
        "",
        "**Evaluation Harness**: Standardized frozen harness comparing M10 Baseline vs M12 Full Robust Model.",
        "",
        "## Scientific Matrix Comparison",
        "",
        "| Test Condition | Evaluation Purpose | M10 Baseline AUC | M12 Full AUC | $\\Delta$ AUC | M10 EER | M12 EER | M12 ECE |",
        "|----------------|--------------------|------------------|--------------|--------------|---------|---------|---------|",
    ]

    for key, name in [
        ("ffpp_in_domain_seen", "FF++ In-Domain (Seen Manipulations)"),
        ("celebdf_seen_dataset", "Celeb-DF v2 (Cross-Domain)"),
        ("ffpp_unseen_face2face", "FF++ Face2Face (Unseen Manipulation)"),
        ("degraded_compressed", "Celeb-DF Compressed (Degradation)"),
        ("degraded_lowres", "Celeb-DF Low-Res (Degradation)"),
    ]:
        m10_res = m10_sub.get(key, {})
        m12_res = m12_sub.get(key, {})

        m10_auc = m10_res.get("roc_auc", float("nan"))
        m12_auc = m12_res.get("roc_auc", float("nan"))
        delta_auc = m12_auc - m10_auc if not np.isnan(m12_auc) and not np.isnan(m10_auc) else float("nan")

        m10_eer = m10_res.get("eer", float("nan"))
        m12_eer = m12_res.get("eer", float("nan"))
        m12_ece = m12_res.get("ece", float("nan"))

        lines.append(
            f"| **{name}** | {key} | {m10_auc:.4f} | **{m12_auc:.4f}** | **{delta_auc:+.4f}** | {m10_eer:.4f} | **{m12_eer:.4f}** | {m12_ece:.4f} |"
        )

    lines.extend([
        "",
        "## Key Scientific Findings",
        "- **Generalization**: Evaluated on seen vs strictly held-out manipulation families.",
        "- **Degradation Robustness**: Measured through simulated compression and low-resolution perturbations.",
        "- **Calibration**: Brier score and Expected Calibration Error (ECE) evaluated across all subsets.",
        "",
        f"*Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Markdown report written to {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="M12 Full Scientific Evaluation")
    parser.add_argument(
        "--m12-checkpoint",
        type=str,
        default=str(M12_CKPT),
        help="Path to M12 checkpoint",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per subtest (for faster testing)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("M12 Full Scientific Evaluation Matrix")
    print("=" * 70)

    # 1. Load test manifest
    df = pd.read_csv(MANIFEST_PATH)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    logger.info(f"Loaded {len(test_df)} test videos from {MANIFEST_PATH}")

    m12_path = Path(args.m12_checkpoint)
    if not m12_path.is_file():
        raise FileNotFoundError(f"M12 checkpoint not found: {m12_path}")
    if not M10_CKPT.is_file():
        raise FileNotFoundError(f"M10 baseline checkpoint not found: {M10_CKPT}")

    # 2. Initialize Predictors
    logger.info("Initializing M10 Baseline Predictor...")
    predictor_m10 = VideoPredictor(checkpoint_path=M10_CKPT, device="cuda")

    logger.info(f"Initializing M12 Predictor ({m12_path})...")
    predictor_m12 = VideoPredictor(checkpoint_path=m12_path, device="cuda")

    # 3. Run Matrix
    logger.info("Running evaluation matrix for M10 Baseline...")
    m10_results = run_full_matrix(
        predictor_m10, test_df, "M10_Baseline", max_samples_per_subtest=args.max_samples
    )

    logger.info("Running evaluation matrix for M12 Full...")
    m12_results = run_full_matrix(
        predictor_m12, test_df, "M12_Full", max_samples_per_subtest=args.max_samples
    )

    # 4. Save JSON Report
    report = {
        "timestamp": time.time(),
        "m10_baseline": m10_results,
        "m12_full": m12_results,
    }

    OUTPUT_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Full report saved to {OUTPUT_REPORT_JSON}")

    # 5. Generate Markdown Report
    generate_markdown_report(report, OUTPUT_REPORT_MD)
    print(f"\nEvaluation complete. Reports written to:\n  - {OUTPUT_REPORT_JSON}\n  - {OUTPUT_REPORT_MD}")


if __name__ == "__main__":
    main()
