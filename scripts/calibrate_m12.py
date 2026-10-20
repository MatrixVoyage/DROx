"""M12 Calibration and OOD Reference Fitting Script.

Strictly uses the VALIDATION split to:
1. Extract in-distribution penultimate temporal feature representations.
2. Fit OODFeatureEstimator centroid and standardized covariance.
3. Compute MC-Dropout epistemic variance distribution on clean validation data.
4. Calibrate safety thresholds for the three-state (REAL, FAKE, REJECT) decision policy.
5. Save ood_reference.json and calibration_config.json alongside checkpoint.

Usage:
    .venv\\Scripts\\python.exe scripts/calibrate_m12.py [--checkpoint checkpoints/m12_pilot/best.pth] [--split val]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ProcessedVideoDataset
from src.data.dataset import ProcessedVideoDataset
from src.evaluation.calibration import TemperatureScaler, compute_calibration_metrics
from src.inference.uncertainty import MCDropoutEstimator, OODFeatureEstimator
from src.models.detector import DeepfakeDetector
from src.utils.config import _dict_to_config, load_config
from src.utils.logger import setup_logger

logger = setup_logger("m12_calibration")


def calibrate(
    checkpoint_path: Path,
    manifest_path: Path,
    processed_dir: Path,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    """Fit TemperatureScaler, OOD reference, and calibrate uncertainty thresholds on validation split."""
    logger.info(f"Loading M12 full checkpoint from {checkpoint_path}")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = ckpt.get("config", {})

    if config_dict:
        config = _dict_to_config(config_dict)
    else:
        config = load_config("configs/experiments/m12_full.yaml")

    model = DeepfakeDetector(config.model)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load validation dataset
    df = pd.read_csv(manifest_path)
    val_dataset = ProcessedVideoDataset(
        manifest=df,
        split="val",
        processed_dir=processed_dir,
        expected_num_frames=config.data.sampling.num_frames,
    )
    logger.info(f"Loaded {len(val_dataset)} validation samples for calibration.")

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    features_list: List[np.ndarray] = []
    logits_list: List[float] = []
    labels_list: List[int] = []
    logger.info("Extracting validation feature embeddings and raw logits...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs, targets = batch[0].to(device), batch[1]
            logits, temporal_features = model(inputs, return_embeddings=True)
            feat_np = temporal_features.cpu().numpy()
            features_list.append(feat_np)
            logits_list.extend(logits.squeeze(-1).cpu().numpy().tolist())
            labels_list.extend(targets.numpy().tolist())

    all_features = np.concatenate(features_list, axis=0)
    all_logits = np.array(logits_list, dtype=np.float32)
    all_labels = np.array(labels_list, dtype=np.int64)
    logger.info(f"Extracted features shape: {all_features.shape}, logits shape: {all_logits.shape}")

    # 1. Temperature Scaling
    logger.info("Fitting post-hoc TemperatureScaler on validation logits...")
    raw_probs = 1.0 / (1.0 + np.exp(-all_logits))
    pre_calib_metrics = compute_calibration_metrics(raw_probs, all_labels)
    logger.info(f"Pre-calibration Val ECE: {pre_calib_metrics.ece:.4f}, Brier: {pre_calib_metrics.brier_score:.4f}, NLL: {pre_calib_metrics.nll:.4f}")

    scaler = TemperatureScaler(init_temperature=1.0)
    opt_temp = scaler.fit(torch.from_numpy(all_logits), torch.from_numpy(all_labels), max_iter=100)
    calibrated_probs = 1.0 / (1.0 + np.exp(-all_logits / opt_temp))
    post_calib_metrics = compute_calibration_metrics(calibrated_probs, all_labels)
    logger.info(f"Post-calibration Val (T={opt_temp:.4f}) ECE: {post_calib_metrics.ece:.4f}, Brier: {post_calib_metrics.brier_score:.4f}, NLL: {post_calib_metrics.nll:.4f}")

    # 2. Fit OOD Feature Estimator
    feature_dim = all_features.shape[1]
    ood_estimator = OODFeatureEstimator(feature_dim=feature_dim)
    ood_estimator.fit(all_features, std_threshold=3.5)

    ood_ref_path = checkpoint_path.parent / "ood_reference.json"
    ood_estimator.save(ood_ref_path)

    # Compute validation distance distribution
    val_distances = [ood_estimator.compute_distance(f) for f in all_features]
    p95_distance = float(np.percentile(val_distances, 95))
    p99_distance = float(np.percentile(val_distances, 99))
    logger.info(f"In-distribution distance: mean={np.mean(val_distances):.2f}, 95th%={p95_distance:.2f}, 99th%={p99_distance:.2f}")

    # 3. Estimate Epistemic Uncertainty via MC-Dropout
    logger.info("Estimating epistemic uncertainty distribution via MC-Dropout on 100 validation samples...")
    mc_estimator = MCDropoutEstimator(model=model, num_samples=10)
    mc_variances = []

    # Stratified validation sample for speed and class balance
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    real_indices = val_df[val_df["label"] == 0].index[:50].tolist()
    fake_indices = val_df[val_df["label"] == 1].index[:50].tolist()
    eval_indices = real_indices + fake_indices

    for idx in eval_indices:
        if idx < len(val_dataset):
            sample_tensor = val_dataset[idx][0].unsqueeze(0).to(device)
            _, var_p, _, _ = mc_estimator.estimate(sample_tensor, device=device)
            mc_variances.append(var_p)

    p95_variance = float(np.percentile(mc_variances, 95)) if mc_variances else 0.001
    p99_variance = float(np.percentile(mc_variances, 99)) if mc_variances else 0.002
    logger.info(f"In-distribution variance: mean={np.mean(mc_variances):.6f}, 95th%={p95_variance:.6f}, 99th%={p99_variance:.6f}")

    # 4. Save Calibration Configuration
    calib_config = {
        "calibration_version": "m12_full_validation",
        "calibration_dataset": "M12 Full Validation Split (mixed FF++ and Celeb-DF)",
        "calibration_samples": len(all_features),
        "temperature": round(opt_temp, 4),
        "pre_calibration_metrics": {
            "ece": pre_calib_metrics.ece,
            "brier": pre_calib_metrics.brier_score,
            "nll": pre_calib_metrics.nll,
        },
        "post_calibration_metrics": {
            "ece": post_calib_metrics.ece,
            "brier": post_calib_metrics.brier_score,
            "nll": post_calib_metrics.nll,
        },
        "classification_threshold": 0.50,
        "max_uncertainty_variance": round(max(0.04, p99_variance * 2.0), 4),
        "ood_distance_threshold": round(max(3.0, p99_distance * 1.1), 2),
        "val_distance_p95": round(p95_distance, 4),
        "val_distance_p99": round(p99_distance, 4),
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    calib_path = checkpoint_path.parent / "calibration_config.json"
    with open(calib_path, "w") as f:
        json.dump(calib_config, f, indent=2)

    logger.info(f"Calibration configuration saved to {calib_path}")
    print(f"\nCalibration complete:\n  - Optimal Temperature T: {opt_temp:.4f}")
    print(f"  - ECE reduction: {pre_calib_metrics.ece:.4f} -> {post_calib_metrics.ece:.4f}")
    print(f"  - OOD Distance threshold (99th% * 1.1): {calib_config['ood_distance_threshold']}")
    print(f"  - Max Epistemic Variance threshold: {calib_config['max_uncertainty_variance']}")
    print(f"  - Saved to {calib_path} & {ood_ref_path}")
    return calib_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate M12 Full model on validation set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/m12_full/best.pth",
        help="Path to trained checkpoint",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="manifests/mixed_robust_full.csv",
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="processed/m12_full",
        help="Path to processed data directory",
    )
    args = parser.parse_args()

    calibrate(
        checkpoint_path=Path(args.checkpoint),
        manifest_path=Path(args.manifest),
        processed_dir=Path(args.processed_dir),
    )


if __name__ == "__main__":
    main()

