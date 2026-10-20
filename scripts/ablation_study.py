"""M12 Scientific Ablation Study across Six Rejection & Architecture Configurations.

Configurations Evaluated:
1. M10 Baseline: Original baseline (uncalibrated, no rejection)
2. M12 Full: Robust training improvement (temperature-scaled, no rejection)
3. M12 + Quality: Quality gating enabled (rejection on low detection / high fallback frames)
4. M12 + OOD: Feature-distance OOD detection enabled (standardized distance > 3.0)
5. M12 + MC Dropout: Epistemic uncertainty rejection enabled (variance > 0.04)
6. M12 + All: Complete calibrated system (Quality + OOD + MC Dropout + Temperature Scaling)

Output:
    reports/m12_ablation_study.json
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
import torch.nn as nn

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
from src.utils.config import _dict_to_config
from src.utils.logger import setup_logger

logger = setup_logger("m12_ablation_study")

MANIFEST_PATH = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"
M10_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "m10_selected_candidate" / "best.pth"
M12_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "best.pth"
OOD_REF_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "ood_reference.json"
CALIB_CONFIG_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "calibration_config.json"
OUTPUT_JSON = PROJECT_ROOT / "reports" / "m12_ablation_study.json"


def load_m10_model(device: torch.device) -> DeepfakeDetector:
    """Load frozen M10 baseline candidate model."""
    logger.info(f"Loading M10 baseline model from {M10_CHECKPOINT_PATH}")
    ckpt = torch.load(M10_CHECKPOINT_PATH, map_location=device, weights_only=False)
    config = _dict_to_config(ckpt["config"])
    model = DeepfakeDetector(config.model)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def load_m12_pipeline(device: torch.device):
    """Load M12 model and calibration components."""
    logger.info(f"Loading M12 full model from {M12_CHECKPOINT_PATH}")
    ckpt = torch.load(M12_CHECKPOINT_PATH, map_location=device, weights_only=False)
    config = _dict_to_config(ckpt["config"])
    model = DeepfakeDetector(config.model)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    with open(CALIB_CONFIG_PATH, "r") as f:
        calib_cfg = json.load(f)

    temp = float(calib_cfg.get("temperature", 1.6369))
    max_var = float(calib_cfg.get("max_uncertainty_variance", 0.04))
    ood_thresh = float(calib_cfg.get("ood_distance_threshold", 3.0))

    mc_estimator = MCDropoutEstimator(model=model, num_samples=10, temperature=temp)

    ood_estimator = OODFeatureEstimator(feature_dim=1280)
    ood_estimator.load(OOD_REF_PATH)

    quality_assessor = QualityAssessor(max_fallback_ratio=0.50, min_detected_frames=8)

    decision_policy = CalibratedDecisionPolicy(
        classification_threshold=0.50,
        max_uncertainty_variance=max_var,
        ood_distance_threshold=ood_thresh,
    )

    return model, mc_estimator, ood_estimator, quality_assessor, decision_policy, temp


def evaluate_config(
    config_name: str,
    samples: List[Dict[str, Any]],
    m10_model: DeepfakeDetector,
    m12_model: DeepfakeDetector,
    mc_estimator: MCDropoutEstimator,
    ood_estimator: OODFeatureEstimator,
    quality_assessor: QualityAssessor,
    decision_policy: CalibratedDecisionPolicy,
    temperature: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate a specific ablation configuration across benchmark samples."""
    logger.info(f"Running Ablation Evaluation for: {config_name} ({len(samples)} samples)...")
    records = []

    for item in samples:
        tensor = item["tensor"].to(device)
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(0)
        true_label = item["label"]
        category = item["category"]  # in_domain, cross_dataset, unseen_manipulation, degraded, synthetic_ood
        is_synthetic = item.get("is_synthetic", False)

        # Quality assessment
        if is_synthetic:
            quality = quality_assessor.assess(
                total_frames=16, detected_frames=0, previous_bbox_frames=0, center_crop_frames=16, failed_frames=0
            )
        else:
            meta_cnt = item.get("counters", {})
            quality = quality_assessor.assess(
                total_frames=16,
                detected_frames=meta_cnt.get("detected_frames", 16),
                previous_bbox_frames=meta_cnt.get("previous_bbox_frames", 0),
                center_crop_frames=meta_cnt.get("center_crop_frames", 0),
                failed_frames=meta_cnt.get("failed_frames", 0),
            )

        if config_name == "M10":
            # Baseline: M10 checkpoint, unscaled sigmoid, zero rejection
            with torch.no_grad():
                logits = m10_model(tensor)
                p = float(torch.sigmoid(logits).squeeze().item())
            is_rejected = False
            pred_label = 1 if p >= 0.5 else 0

        elif config_name == "M12 full":
            # M12 full: Temperature scaled, zero rejection
            with torch.no_grad():
                logits = m12_model(tensor)
                p = float(torch.sigmoid(logits / temperature).squeeze().item())
            is_rejected = False
            pred_label = 1 if p >= 0.5 else 0

        elif config_name == "M12 + quality rejection":
            # Quality rejection only
            if not quality.is_acceptable:
                is_rejected = True
                pred_label = None
                p = 0.5
            else:
                is_rejected = False
                with torch.no_grad():
                    logits = m12_model(tensor)
                    p = float(torch.sigmoid(logits / temperature).squeeze().item())
                pred_label = 1 if p >= 0.5 else 0

        elif config_name == "M12 + OOD":
            # OOD distance rejection only
            with torch.no_grad():
                logits, feats = m12_model(tensor, return_embeddings=True)
                p = float(torch.sigmoid(logits / temperature).squeeze().item())
                dist = ood_estimator.compute_distance(feats.squeeze(0).cpu())
            if dist > decision_policy.ood_distance_threshold:
                is_rejected = True
                pred_label = None
            else:
                is_rejected = False
                pred_label = 1 if p >= 0.5 else 0

        elif config_name == "M12 + MC Dropout":
            # MC Dropout uncertainty rejection only
            mean_p, var_p, entropy, feats = mc_estimator.estimate(tensor, device=device)
            p = mean_p
            if var_p > decision_policy.max_uncertainty_variance:
                is_rejected = True
                pred_label = None
            else:
                is_rejected = False
                pred_label = 1 if p >= 0.5 else 0

        elif config_name == "M12 + all":
            # Full calibrated system
            mean_p, var_p, entropy, feats = mc_estimator.estimate(tensor, device=device)
            dist = ood_estimator.compute_distance(feats)
            unc_signals = UncertaintySignals(
                mean_probability=mean_p,
                epistemic_variance=var_p,
                entropy=entropy,
                mc_samples=10,
                ood_distance=dist,
                is_ood=dist > decision_policy.ood_distance_threshold,
                is_uncertain=var_p > decision_policy.max_uncertainty_variance,
            )
            verdict, conf, code, reason = decision_policy.decide(mean_p, quality, unc_signals)
            is_rejected = verdict == "Reject"
            p = mean_p
            pred_label = 1 if verdict == "Deepfake" else (0 if verdict == "Real" else None)

        else:
            raise ValueError(f"Unknown configuration: {config_name}")

        records.append({
            "config": config_name,
            "category": category,
            "true_label": true_label,
            "pred_label": pred_label,
            "probability": p,
            "is_rejected": is_rejected,
            "is_synthetic": is_synthetic,
        })

    df = pd.DataFrame(records)
    total_n = len(df)
    acc_df = df[~df["is_rejected"]].reset_index(drop=True)
    rej_df = df[df["is_rejected"]].reset_index(drop=True)

    coverage = len(acc_df) / total_n if total_n > 0 else 0.0
    overall_rej_rate = len(rej_df) / total_n if total_n > 0 else 0.0

    # Selective Accuracy on accepted face samples (excluding synthetic OOD)
    face_accepted = acc_df[acc_df["true_label"].isin([0, 1])]
    if len(face_accepted) > 0:
        overall_sel_acc = float(accuracy_score(face_accepted["true_label"], face_accepted["pred_label"]))
    else:
        overall_sel_acc = 0.0

    # Clean Real Rejection Rate (FRR)
    clean_real = df[(df["category"] == "in_domain") & (df["true_label"] == 0)]
    frr = float(clean_real["is_rejected"].mean()) if len(clean_real) > 0 else 0.0

    # Synthetic OOD True Rejection Rate (TRR)
    synth_ood = df[df["is_synthetic"]]
    trr = float(synth_ood["is_rejected"].mean()) if len(synth_ood) > 0 else 0.0

    # Category accuracies on accepted samples
    def get_cat_acc(cat_name: str) -> Optional[float]:
        sub = df[(df["category"] == cat_name) & (~df["is_rejected"]) & (df["true_label"].isin([0, 1]))]
        if len(sub) > 0:
            return float(accuracy_score(sub["true_label"], sub["pred_label"]))
        return None

    in_domain_acc = get_cat_acc("in_domain")
    cross_dataset_acc = get_cat_acc("cross_dataset")
    unseen_manip_acc = get_cat_acc("unseen_manipulation")
    degraded_acc = get_cat_acc("degraded")

    return {
        "configuration": config_name,
        "coverage": round(coverage, 4),
        "overall_selective_accuracy": round(overall_sel_acc, 4),
        "in_domain_accuracy": round(in_domain_acc, 4) if in_domain_acc is not None else None,
        "cross_dataset_accuracy": round(cross_dataset_acc, 4) if cross_dataset_acc is not None else None,
        "unseen_manipulation_accuracy": round(unseen_manip_acc, 4) if unseen_manip_acc is not None else None,
        "degradation_accuracy": round(degraded_acc, 4) if degraded_acc is not None else None,
        "clean_real_frr": round(frr, 4),
        "synthetic_ood_trr": round(trr, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="M12 Scientific Ablation Study")
    parser.add_argument("--samples-per-slice", type=int, default=30, help="Samples per slice")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Executing ablation study on {device}")

    # Load models
    m10_model = load_m10_model(device)
    m12_model, mc_estimator, ood_estimator, quality_assessor, decision_policy, temp = load_m12_pipeline(device)

    # Prepare Benchmark Dataset
    manifest_df = pd.read_csv(MANIFEST_PATH)
    test_df = manifest_df[manifest_df["split"] == "test"].reset_index(drop=True)
    tensors_dir = PROJECT_ROOT / "processed" / "m12_full" / "test" / "tensors"
    meta_dir = PROJECT_ROOT / "processed" / "m12_full" / "test" / "metadata"
    test_df = test_df[test_df["video_id"].apply(lambda vid: (tensors_dir / f"{vid}.pt").is_file())].reset_index(drop=True)

    n = args.samples_per_slice
    jpeg_aug = RobustAugmentor(
        config=RobustAugmentorConfig(jpeg_p=1.0, blur_p=0.0, noise_p=0.0, resize_p=0.0, reencode_p=0.0, frame_drop_p=0.0),
        seed=42,
    )

    benchmark_samples = []

    # 1. In-Domain (FF++ seen: original + Deepfakes/FaceSwap/NeuralTextures)
    ffpp_clean = test_df[(test_df["dataset"] == "ffpp") & (test_df["manipulation_type"] != "Face2Face")].sample(
        n=min(n, len(test_df[(test_df["dataset"] == "ffpp") & (test_df["manipulation_type"] != "Face2Face")])),
        random_state=42
    )
    for _, row in ffpp_clean.iterrows():
        vid = row["video_id"]
        t = torch.load(tensors_dir / f"{vid}.pt", map_location="cpu", weights_only=False)
        cnt = {}
        if (meta_dir / f"{vid}.json").is_file():
            with open(meta_dir / f"{vid}.json") as mf:
                cnt = json.load(mf).get("counters", {})
        benchmark_samples.append({
            "tensor": t, "label": int(row["label"]), "category": "in_domain", "video_id": vid, "counters": cnt, "is_synthetic": False
        })

    # 2. Cross-Dataset (Celeb-DF pristine + deepfake)
    celeb_df = test_df[test_df["dataset"] == "celebdf"].sample(
        n=min(n, (test_df["dataset"] == "celebdf").sum()), random_state=42
    )
    for _, row in celeb_df.iterrows():
        vid = row["video_id"]
        t = torch.load(tensors_dir / f"{vid}.pt", map_location="cpu", weights_only=False)
        cnt = {}
        if (meta_dir / f"{vid}.json").is_file():
            with open(meta_dir / f"{vid}.json") as mf:
                cnt = json.load(mf).get("counters", {})
        benchmark_samples.append({
            "tensor": t, "label": int(row["label"]), "category": "cross_dataset", "video_id": vid, "counters": cnt, "is_synthetic": False
        })

    # 3. Unseen Manipulation Family (Held-out FF++ Face2Face)
    f2f_df = test_df[test_df["manipulation_type"] == "Face2Face"].sample(
        n=min(n, (test_df["manipulation_type"] == "Face2Face").sum()), random_state=42
    )
    for _, row in f2f_df.iterrows():
        vid = row["video_id"]
        t = torch.load(tensors_dir / f"{vid}.pt", map_location="cpu", weights_only=False)
        cnt = {}
        if (meta_dir / f"{vid}.json").is_file():
            with open(meta_dir / f"{vid}.json") as mf:
                cnt = json.load(mf).get("counters", {})
        benchmark_samples.append({
            "tensor": t, "label": int(row["label"]), "category": "unseen_manipulation", "video_id": vid, "counters": cnt, "is_synthetic": False
        })

    # 4. Degradation Robustness (JPEG compressed in-domain samples)
    for s in benchmark_samples[:n]:
        t_aug, _ = jpeg_aug.augment_tensor(s["tensor"].clone())
        benchmark_samples.append({
            "tensor": t_aug, "label": s["label"], "category": "degraded", "video_id": f"deg_{s['video_id']}", "counters": s["counters"], "is_synthetic": False
        })

    # 5. Synthetic Non-Face OOD (Extreme Gaussian Noise)
    rng = np.random.RandomState(42)
    for i in range(n):
        noise_t = torch.from_numpy(rng.randn(16, 3, 112, 112).astype(np.float32))
        benchmark_samples.append({
            "tensor": noise_t, "label": -1, "category": "synthetic_ood", "video_id": f"noise_{i}", "counters": {}, "is_synthetic": True
        })

    logger.info(f"Total benchmark samples compiled: {len(benchmark_samples)}")

    configs = [
        "M10",
        "M12 full",
        "M12 + quality rejection",
        "M12 + OOD",
        "M12 + MC Dropout",
        "M12 + all",
    ]

    ablation_results = []
    for cfg in configs:
        res = evaluate_config(
            config_name=cfg,
            samples=benchmark_samples,
            m10_model=m10_model,
            m12_model=m12_model,
            mc_estimator=mc_estimator,
            ood_estimator=ood_estimator,
            quality_assessor=quality_assessor,
            decision_policy=decision_policy,
            temperature=temp,
            device=device,
        )
        ablation_results.append(res)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(ablation_results, f, indent=2)

    logger.info(f"Ablation results saved to {OUTPUT_JSON}")

    # Print Table
    print("\n" + "=" * 105)
    print("M12 SYSTEMATIC ABLATION MATRIX")
    print("=" * 105)
    header = f"{'Configuration':<25} | {'Coverage':<9} | {'Sel. Acc':<9} | {'In-Domain':<10} | {'Cross-DS':<9} | {'Face2Face':<9} | {'Degraded':<9} | {'OOD TRR':<8}"
    print(header)
    print("-" * 105)
    for r in ablation_results:
        cov_str = f"{r['coverage']*100:.1f}%"
        sel_str = f"{r['overall_selective_accuracy']*100:.1f}%"
        id_str = f"{r['in_domain_accuracy']*100:.1f}%" if r['in_domain_accuracy'] is not None else "N/A"
        cd_str = f"{r['cross_dataset_accuracy']*100:.1f}%" if r['cross_dataset_accuracy'] is not None else "N/A"
        f2f_str = f"{r['unseen_manipulation_accuracy']*100:.1f}%" if r['unseen_manipulation_accuracy'] is not None else "N/A"
        deg_str = f"{r['degradation_accuracy']*100:.1f}%" if r['degradation_accuracy'] is not None else "N/A"
        trr_str = f"{r['synthetic_ood_trr']*100:.1f}%"
        print(f"{r['configuration']:<25} | {cov_str:<9} | {sel_str:<9} | {id_str:<10} | {cd_str:<9} | {f2f_str:<9} | {deg_str:<9} | {trr_str:<8}")
    print("=" * 105)


if __name__ == "__main__":
    main()
