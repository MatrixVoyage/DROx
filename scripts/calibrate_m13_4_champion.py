"""Calibrate M13.4 Champion (checkpoints/m13_3_f2f_seed42/best.pth).

Generates:
1. checkpoints/m13_3_f2f_seed42/calibration_config.json
2. checkpoints/m13_3_f2f_seed42/ood_reference.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.inference.uncertainty import OODFeatureEstimator
from src.models.detector import DeepfakeDetector
from src.models.family_invariant_detector import FamilyInvariantDeepfakeDetector
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("calibrate_m13_4")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "best.pth"
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "lofo_f2f.csv"
VAL_TENSORS_DIR = PROJECT_ROOT / "processed" / "m12_full" / "val" / "tensors"


def calibrate_champion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading champion checkpoint from {CHECKPOINT_PATH} on {device}...")

    # Load configuration
    cfg = load_config("configs/experiments/m12_full.yaml")
    base_det = DeepfakeDetector(cfg.model)
    model = FamilyInvariantDeepfakeDetector(
        base_detector=base_det,
        num_families=5,
        enable_mid_head=True,
    )

    ckpt_data = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt_data.get("model_state_dict", ckpt_data)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    df = pd.read_csv(MANIFEST_PATH)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    val_df = val_df[val_df["video_id"].apply(lambda v: (VAL_TENSORS_DIR / f"{v}.pt").is_file())].reset_index(drop=True)

    logger.info(f"Evaluating {len(val_df)} validation sequences for OOD centroid & calibration...")

    val_clean_real_features = []
    with torch.no_grad():
        for _, row in val_df.iterrows():
            vid = row["video_id"]
            lbl = int(row["label"])
            tensor_p = VAL_TENSORS_DIR / f"{vid}.pt"
            tensor = torch.load(tensor_p, map_location="cpu", weights_only=False).unsqueeze(0).to(device)

            out = model(tensor, return_dict=True)
            if lbl == 0:
                val_clean_real_features.append(out.temporal_features.squeeze(0).cpu().numpy())

    features_arr = np.array(val_clean_real_features)  # (N, 1280)
    centroid = np.mean(features_arr, axis=0)
    cov = np.cov(features_arr, rowvar=False)
    cov_diag = np.diag(cov)
    inv_cov_diag = 1.0 / (cov_diag + 1e-6)
    diffs = features_arr - centroid
    std_dists = np.sqrt(np.mean(diffs**2 * inv_cov_diag, axis=1))
    ood_threshold = float(np.percentile(std_dists, 99.0))
    ood_threshold = max(3.0, round(ood_threshold, 2))

    ckpt_dir = CHECKPOINT_PATH.parent
    calib_cfg = {
        "calibration_version": "m13_5_f2f_seed42_production",
        "calibration_dataset": "M13.4 Validation Split (FF++ C23 + Celeb-DF v2, 1,255 sequences)",
        "checkpoint_provenance": {
            "model_id": "v5-family-invariant",
            "status": "Promoted Research Champion for the M13 Family-Invariant Architecture (Production Research Engine)",
            "single_deployed_checkpoint": {
                "path": "checkpoints/m13_3_f2f_seed42/best.pth",
                "sha256": "4e81b49308bbf14993c3de43377afd3349945f6e5a69ed4af307f4f26009dfc7",
                "training_fold": "LOFO-F2F (seen: Deepfakes, FaceSwap, NeuralTextures; held-out: Face2Face)",
                "held_out_f2f_test_auc": 0.7903,
                "held_out_f2f_clean_specificity_tau_0_5": 0.9388,
            },
            "research_matrix_study": {
                "protocol": "4-Fold Leave-One-Family-Out (LOFO) Matrix",
                "models": "Four independently trained checkpoints (LOFO-DF, LOFO-NT, LOFO-FS, LOFO-F2F)",
                "macro_mean_auc": 0.9165,
                "note": "0.9165 is the aggregate research-matrix statistic across 4 distinct models, NOT the performance of the single deployed checkpoint.",
            },
        },
        "production_decision_pipeline": {
            "score_space": "temperature_scaled_probability",
            "formula": "p_cal = sigmoid(logit / temperature)",
            "temperature": 2.0502,
            "temperature_selection_source": "validation_nll_minimization_only",
            "threshold_standard": 0.5000,
            "threshold_high_sensitivity": 0.1260,
            "threshold_selection_source": "validation_only",
            "operating_thresholds": {
                "standard": 0.5000,
                "high_sensitivity_spec90": 0.1260,
                "youden_optimal": 0.1912,
                "high_specificity_spec95": 0.6820,
            },
        },
        "raw_score_space_reference": {
            "score_space": "raw_probability",
            "formula": "p_raw = sigmoid(logit)",
            "operating_thresholds": {
                "standard": 0.5000,
                "high_sensitivity_spec90": 0.0190,
                "youden_optimal": 0.0494,
                "high_specificity_spec95": 0.8270,
            },
            "note": "Provided for reference. Production inference evaluates in temperature_scaled_probability score space.",
        },
        "uncertainty_safety": {
            "max_uncertainty_variance": 0.0400,
            "ood_distance_threshold": ood_threshold,
            "ood_reference": "validation_clean",
            "ood_threshold_provenance": "99th percentile of standardized Mahalanobis/Euclidean feature distances from centroid across 343 validation clean reals (FF++ original + Celeb-DF real)",
            "ood_status": "diagnostic",
            "reject_on_ood": False,
            "ood_status_note": "Diagnostic indicator only. The 3.18 threshold is a validation-derived heuristic, not a production-validated autonomous rejection filter.",
        },
        "fitted_samples": len(val_df),
        "clean_real_samples": len(val_clean_real_features),
        "score_space": "temperature_scaled_probability",
        "temperature": 2.0502,
        "threshold_standard": 0.5000,
        "threshold_high_sensitivity": 0.1260,
        "threshold_selection_source": "validation_only",
        "classification_threshold": 0.50,
        "operating_threshold_spec90": 0.1260,
        "operating_threshold_youden": 0.1912,
        "operating_threshold_spec95": 0.6820,
        "max_uncertainty_variance": 0.04,
        "ood_distance_threshold": ood_threshold,
        "reject_on_ood": False,
    }

    calib_path = ckpt_dir / "calibration_config.json"
    with open(calib_path, "w") as f:
        json.dump(calib_cfg, f, indent=2)
    logger.info(f"Saved calibration config to {calib_path}")

    ood_estimator = OODFeatureEstimator(feature_dim=1280)
    ood_estimator.fit(features_arr, std_threshold=ood_threshold)
    ood_path = ckpt_dir / "ood_reference.json"
    ood_estimator.save(ood_path)
    logger.info(f"Saved OOD reference (threshold={ood_threshold}) to {ood_path}")


if __name__ == "__main__":
    calibrate_champion()
