"""M12.4 Rejection & Uncertainty Scientific Audit Script.

Computes the empirical operating characteristics of the calibrated three-state decision policy:
1. Coverage, Selective Accuracy, Overall Rejection Rate
2. Clean False Rejection Rate (FRR) and OOD True Rejection Rate (TRR)
3. OOD AUROC (feature distance discrimination on in-distribution vs synthetic OOD noise)
4. Slice-by-slice rejection breakdown:
   - Clean Real
   - Clean Fake
   - Degraded Real
   - Degraded Fake
   - Unseen Manipulation (FF++ Face2Face)
   - Synthetic Non-Face / Noise OOD
5. Coverage–Selective Accuracy Curve (95%, 90%, 85%, 80%, 75%, 70%)

Usage:
    .venv\\Scripts\\python.exe scripts/audit_rejection_metrics.py [--samples-per-slice 50]
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
from sklearn.metrics import accuracy_score, roc_auc_score
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augment import RobustAugmentor, RobustAugmentorConfig
from src.inference.uncertainty import (
    CalibratedDecisionPolicy,
    MCDropoutEstimator,
    OODFeatureEstimator,
    QualityAssessor,
    QualitySignals,
    UncertaintySignals,
)
from src.models.detector import DeepfakeDetector
from src.utils.config import _dict_to_config, load_config
from src.utils.logger import setup_logger

logger = setup_logger("m12_rejection_audit")

MANIFEST_PATH = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "best.pth"
OOD_REF_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "ood_reference.json"
CALIB_CONFIG_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "calibration_config.json"
REPORT_JSON = PROJECT_ROOT / "reports" / "m12_rejection_audit.json"
REPORT_MD = PROJECT_ROOT / "reports" / "m12_full_scientific_report.md"


def load_model_and_calibrators(device: torch.device):
    """Load model checkpoint and calibration components."""
    logger.info(f"Loading M12 checkpoint from {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    config = _dict_to_config(ckpt["config"])

    model = DeepfakeDetector(config.model)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    with open(CALIB_CONFIG_PATH, "r") as f:
        calib_cfg = json.load(f)

    temperature = float(calib_cfg.get("temperature", 1.6369))

    # Calibrators with fitted temperature scaling
    mc_estimator = MCDropoutEstimator(model=model, num_samples=10, temperature=temperature)

    ood_estimator = OODFeatureEstimator(feature_dim=1280)
    ood_estimator.load(OOD_REF_PATH)

    decision_policy = CalibratedDecisionPolicy(
        classification_threshold=calib_cfg.get("classification_threshold", 0.50),
        max_uncertainty_variance=calib_cfg.get("max_uncertainty_variance", 0.04),
        ood_distance_threshold=calib_cfg.get("ood_distance_threshold", 3.0),
    )

    quality_assessor = QualityAssessor(max_fallback_ratio=0.50, min_detected_frames=8)

    return model, mc_estimator, ood_estimator, decision_policy, quality_assessor


def evaluate_slice(
    model: DeepfakeDetector,
    mc_estimator: MCDropoutEstimator,
    ood_estimator: OODFeatureEstimator,
    decision_policy: CalibratedDecisionPolicy,
    quality_assessor: QualityAssessor,
    slice_df: pd.DataFrame,
    slice_name: str,
    degradation: Optional[str] = None,
    is_synthetic_ood: bool = False,
    device: torch.device = torch.device("cuda"),
) -> List[Dict[str, Any]]:
    """Evaluate calibrated decision policy on a slice of samples."""
    logger.info(f"Auditing slice: {slice_name} ({len(slice_df)} samples, degradation={degradation})...")

    # Set up degradations if requested
    jpeg_aug = RobustAugmentor(
        config=RobustAugmentorConfig(jpeg_p=1.0, blur_p=0.0, noise_p=0.0, resize_p=0.0, reencode_p=0.0, frame_drop_p=0.0),
        seed=42,
    )
    lowres_aug = RobustAugmentor(
        config=RobustAugmentorConfig(jpeg_p=0.0, blur_p=0.0, noise_p=0.0, resize_p=1.0, reencode_p=0.0, frame_drop_p=0.0),
        seed=42,
    )

    records = []

    for idx, row in slice_df.iterrows():
        vid_id = str(row.get("video_id", f"synthetic_ood_{idx}"))
        lbl = int(row.get("label", -1))

        if is_synthetic_ood:
            # Generate pure Gaussian noise tensor as extreme OOD non-face sequence
            rng = np.random.RandomState(42 + idx)
            noise_tensor = torch.from_numpy(rng.randn(16, 3, 112, 112).astype(np.float32))
            tensor = noise_tensor
            # Synthetic quality signals: zero faces detected
            quality_signals = quality_assessor.assess(
                total_frames=16, detected_frames=0, previous_bbox_frames=0, center_crop_frames=16, failed_frames=0
            )
        else:
            tensor_file = PROJECT_ROOT / "processed" / "m12_full" / "test" / "tensors" / f"{vid_id}.pt"
            if not tensor_file.is_file():
                continue

            tensor = torch.load(tensor_file, map_location="cpu", weights_only=False)

            if degradation == "jpeg_compression":
                tensor, _ = jpeg_aug.augment_tensor(tensor)
            elif degradation == "low_resolution":
                tensor, _ = lowres_aug.augment_tensor(tensor)

            # Load actual saved metadata for quality signals
            meta_file = PROJECT_ROOT / "processed" / "m12_full" / "test" / "metadata" / f"{vid_id}.json"
            if meta_file.is_file():
                with open(meta_file, "r") as mf:
                    meta = json.load(mf)
                cnt = meta.get("counters", {})
                quality_signals = quality_assessor.assess(
                    total_frames=16,
                    detected_frames=cnt.get("detected_frames", 16),
                    previous_bbox_frames=cnt.get("previous_bbox_frames", 0),
                    center_crop_frames=cnt.get("center_crop_frames", 0),
                    failed_frames=cnt.get("failed_frames", 0),
                )
            else:
                quality_signals = quality_assessor.assess(total_frames=16, detected_frames=16)

        # 1. Epistemic uncertainty & probability via MC-Dropout
        mean_p, var_p, entropy, mean_features = mc_estimator.estimate(tensor, device=device)

        # 2. OOD distance
        ood_dist = ood_estimator.compute_distance(mean_features)
        is_ood = ood_estimator.is_ood(ood_dist)

        # 3. Decision policy
        unc_signals = UncertaintySignals(
            mean_probability=mean_p,
            epistemic_variance=var_p,
            entropy=entropy,
            mc_samples=10,
            ood_distance=ood_dist,
            is_ood=is_ood,
            is_uncertain=var_p > decision_policy.max_uncertainty_variance,
        )

        verdict, confidence, code, reason = decision_policy.decide(mean_p, quality_signals, unc_signals)

        records.append({
            "slice": slice_name,
            "video_id": vid_id,
            "true_label": lbl,
            "verdict": verdict,  # "Real", "Deepfake", "Reject"
            "is_rejected": verdict == "Reject",
            "rejection_code": code,
            "rejection_reason": reason,
            "probability": round(mean_p, 4),
            "confidence": round(confidence, 4),
            "epistemic_variance": round(var_p, 6),
            "predictive_entropy": round(entropy, 4),
            "ood_distance": round(ood_dist, 4),
            "quality_acceptable": quality_signals.is_acceptable,
        })

    return records


def compute_audit_summary(all_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute complete scientific audit metrics across all slices."""
    df = pd.DataFrame(all_records)
    total_samples = len(df)
    accepted_df = df[~df["is_rejected"]].reset_index(drop=True)
    rejected_df = df[df["is_rejected"]].reset_index(drop=True)

    coverage = len(accepted_df) / total_samples if total_samples > 0 else 0.0
    overall_rejection_rate = len(rejected_df) / total_samples if total_samples > 0 else 0.0

    # Selective Accuracy on accepted samples where true_label in {0, 1}
    eval_accepted = accepted_df[accepted_df["true_label"].isin([0, 1])].copy()
    if len(eval_accepted) > 0:
        pred_labels = eval_accepted["verdict"].map({"Real": 0, "Deepfake": 1})
        selective_accuracy = float(accuracy_score(eval_accepted["true_label"], pred_labels))
    else:
        selective_accuracy = float("nan")

    # Slice-by-slice metrics
    slice_metrics = {}
    for s_name, group in df.groupby("slice"):
        n_tot = len(group)
        n_rej = int(group["is_rejected"].sum())
        n_acc = n_tot - n_rej
        rej_rate = n_rej / n_tot if n_tot > 0 else 0.0

        group_acc = group[~group["is_rejected"] & group["true_label"].isin([0, 1])]
        if len(group_acc) > 0:
            p_lbls = group_acc["verdict"].map({"Real": 0, "Deepfake": 1})
            s_acc = float(accuracy_score(group_acc["true_label"], p_lbls))
        else:
            s_acc = float("nan")

        slice_metrics[s_name] = {
            "total_samples": n_tot,
            "accepted_samples": n_acc,
            "rejected_samples": n_rej,
            "rejection_rate": round(rej_rate, 4),
            "selective_accuracy": round(s_acc, 4) if not np.isnan(s_acc) else None,
            "mean_ood_distance": round(float(group["ood_distance"].mean()), 4),
            "mean_epistemic_variance": round(float(group["epistemic_variance"].mean()), 6),
        }

    # Clean False Rejection Rate (FRR): Rejection rate on Clean Real
    clean_real_rej = slice_metrics.get("Clean Real (In-Domain & Pristine)", {}).get("rejection_rate", 0.0)
    # OOD True Rejection Rate (TRR): Rejection rate on Synthetic Non-Face OOD
    ood_trr = slice_metrics.get("Synthetic Non-Face OOD (Extreme Noise)", {}).get("rejection_rate", 1.0)

    # OOD AUROC: ability of ood_distance to discriminate Clean In-Distribution vs Synthetic OOD
    in_dist_dists = df[df["slice"].isin(["Clean Real (In-Domain & Pristine)", "In-Domain Seen Fake (FF++)"])]["ood_distance"].values
    ood_dists = df[df["slice"] == "Synthetic Non-Face OOD (Extreme Noise)"]["ood_distance"].values
    if len(in_dist_dists) > 0 and len(ood_dists) > 0:
        y_true_ood = np.concatenate([np.zeros(len(in_dist_dists)), np.ones(len(ood_dists))])
        y_scores_ood = np.concatenate([in_dist_dists, ood_dists])
        ood_auroc = float(roc_auc_score(y_true_ood, y_scores_ood))
    else:
        ood_auroc = float("nan")

    # Coverage-Selective Accuracy Curve
    # Rank all non-OOD samples by confidence (higher confidence = more certain)
    non_ood_df = df[df["true_label"].isin([0, 1])].copy()
    non_ood_df["pred_label"] = (non_ood_df["probability"] >= 0.50).astype(int)
    non_ood_df["is_correct"] = (non_ood_df["pred_label"] == non_ood_df["true_label"]).astype(int)
    # Effective uncertainty score: higher is less confident
    non_ood_df["uncertainty_score"] = non_ood_df["epistemic_variance"] * 100.0 + non_ood_df["ood_distance"]
    sorted_df = non_ood_df.sort_values(by="uncertainty_score", ascending=True).reset_index(drop=True)

    coverage_curve = {}
    for pct in [95, 90, 85, 80, 75, 70]:
        n_keep = max(1, int(round(len(sorted_df) * pct / 100.0)))
        top_k = sorted_df.iloc[:n_keep]
        acc_k = float(top_k["is_correct"].mean())
        coverage_curve[f"{pct}%"] = round(acc_k, 4)

    return {
        "total_queries_evaluated": total_samples,
        "total_accepted": len(accepted_df),
        "total_rejected": len(rejected_df),
        "overall_coverage": round(coverage, 4),
        "overall_selective_accuracy": round(selective_accuracy, 4),
        "overall_rejection_rate": round(overall_rejection_rate, 4),
        "clean_false_rejection_rate_frr": round(clean_real_rej, 4),
        "ood_true_rejection_rate_trr": round(ood_trr, 4),
        "ood_auroc": round(ood_auroc, 4),
        "slice_metrics": slice_metrics,
        "coverage_selective_accuracy_curve": coverage_curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit M12.4 Calibrated Rejection Metrics")
    parser.add_argument("--samples-per-slice", type=int, default=50, help="Samples per slice")
    args = parser.parse_args()

    print("=" * 70)
    print("M12.4 Calibrated Rejection & Uncertainty Scientific Audit")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mc_estimator, ood_estimator, decision_policy, quality_assessor = load_model_and_calibrators(device)

    # Load test manifest and filter to preprocessed test tensors
    df = pd.read_csv(MANIFEST_PATH)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    tensors_dir = PROJECT_ROOT / "processed" / "m12_full" / "test" / "tensors"
    test_df = test_df[test_df["video_id"].apply(lambda vid: (tensors_dir / f"{vid}.pt").is_file())].reset_index(drop=True)
    logger.info(f"Filtered test manifest to {len(test_df)} available preprocessed tensors.")

    n_samples = args.samples_per_slice

    # Define Slices per Scientific Taxonomy:
    # 1. Clean Real (In-domain FF++ original and Celeb-DF pristine)
    clean_real_df = test_df[test_df["label"] == 0].sample(n=min(n_samples, (test_df["label"] == 0).sum()), random_state=42)
    # 2. In-Domain Seen Fake (FF++ Deepfakes, FaceSwap, NeuralTextures)
    in_domain_fake_df = test_df[(test_df["dataset"] == "ffpp") & (test_df["label"] == 1) & (test_df["manipulation_type"] != "Face2Face")].sample(
        n=min(n_samples, len(test_df[(test_df["dataset"] == "ffpp") & (test_df["label"] == 1) & (test_df["manipulation_type"] != "Face2Face")])),
        random_state=42
    )
    # 3. Cross-Dataset Generalization Fake (Celeb-DF deepfake)
    cross_dataset_fake_df = test_df[(test_df["dataset"] == "celebdf") & (test_df["label"] == 1)].sample(
        n=min(n_samples, len(test_df[(test_df["dataset"] == "celebdf") & (test_df["label"] == 1)])),
        random_state=42
    )
    # 4. Degradation Robustness: Degraded Real (Clean Real + JPEG compression)
    degraded_real_df = clean_real_df.copy()
    # 5. Degradation Robustness: Degraded Fake (In-Domain Fake + JPEG compression)
    degraded_fake_df = in_domain_fake_df.copy()
    # 6. Unseen Manipulation-Family Generalization: Held-out FF++ Face2Face
    unseen_df = test_df[test_df["manipulation_type"] == "Face2Face"].sample(
        n=min(n_samples, (test_df["manipulation_type"] == "Face2Face").sum()), random_state=42
    )
    # 7. Synthetic Out-of-Distribution (OOD): Non-Face Extreme Noise
    dummy_ood_df = pd.DataFrame([{"video_id": f"noise_{i}", "label": -1} for i in range(n_samples)])

    all_records = []

    # Run Slices
    all_records.extend(evaluate_slice(model, mc_estimator, ood_estimator, decision_policy, quality_assessor, clean_real_df, "Clean Real (In-Domain & Pristine)", device=device))
    all_records.extend(evaluate_slice(model, mc_estimator, ood_estimator, decision_policy, quality_assessor, in_domain_fake_df, "In-Domain Seen Fake (FF++)", device=device))
    all_records.extend(evaluate_slice(model, mc_estimator, ood_estimator, decision_policy, quality_assessor, cross_dataset_fake_df, "Cross-Dataset Fake (Celeb-DF)", device=device))
    all_records.extend(evaluate_slice(model, mc_estimator, ood_estimator, decision_policy, quality_assessor, degraded_real_df, "Degraded Real (JPEG Robustness)", degradation="jpeg_compression", device=device))
    all_records.extend(evaluate_slice(model, mc_estimator, ood_estimator, decision_policy, quality_assessor, degraded_fake_df, "Degraded Fake (JPEG Robustness)", degradation="jpeg_compression", device=device))
    all_records.extend(evaluate_slice(model, mc_estimator, ood_estimator, decision_policy, quality_assessor, unseen_df, "Unseen Manipulation Family (Held-out Face2Face)", device=device))
    all_records.extend(evaluate_slice(model, mc_estimator, ood_estimator, decision_policy, quality_assessor, dummy_ood_df, "Synthetic Non-Face OOD (Extreme Noise)", is_synthetic_ood=True, device=device))

    # Compute Summary
    audit_summary = compute_audit_summary(all_records)

    # Save to JSON
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w") as f:
        json.dump(audit_summary, f, indent=2)

    logger.info(f"Audit summary written to {REPORT_JSON}")

    # Print summary table
    print("\n" + "=" * 70)
    print("M12.4 REJECTION OPERATING CHARACTERISTICS")
    print("=" * 70)
    print(f"Overall Coverage:               {audit_summary['overall_coverage']*100:.2f}%")
    print(f"Overall Selective Accuracy:     {audit_summary['overall_selective_accuracy']*100:.2f}%")
    print(f"Clean False Rejection Rate (FRR): {audit_summary['clean_false_rejection_rate_frr']*100:.2f}%")
    print(f"OOD True Rejection Rate (TRR):    {audit_summary['ood_true_rejection_rate_trr']*100:.2f}%")
    print(f"OOD Discrimination AUROC:       {audit_summary['ood_auroc']:.4f}")

    print("\nSlice Breakdown:")
    for s_name, metrics in audit_summary["slice_metrics"].items():
        sel_acc_str = f"{metrics['selective_accuracy']*100:.1f}%" if metrics['selective_accuracy'] is not None else "N/A"
        print(f"  - {s_name:<30} | Rej Rate: {metrics['rejection_rate']*100:>5.1f}% | Selective Acc: {sel_acc_str:>6} | Mean OOD Dist: {metrics['mean_ood_distance']:.2f}")

    print("\nCoverage–Selective Accuracy Curve:")
    for pct, acc in audit_summary["coverage_selective_accuracy_curve"].items():
        print(f"  - Coverage {pct:<4} -> Selective Accuracy: {acc*100:.2f}%")

    print("\nAudit successfully completed.")


if __name__ == "__main__":
    main()
