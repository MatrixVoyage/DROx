"""Milestone 22: Calibrate Multi-Model Decision-Level Evidence Fusion.

Zero Neural Network Retraining:
Evaluates frozen Champion M13.5 and Baseline M18 Dual-Stream across the
FF++ C23 validation partition (N=1,255 clips).
Fits 4 evidence fusion strategies and derives locked decision threshold policies
(tau_95*, tau_99*, tau_Youden*).
Saves configuration to checkpoints/m22_ensemble/fusion_config.json.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detector import DeepfakeDetector
from src.models.family_invariant_detector import FamilyInvariantDeepfakeDetector
from src.models.m17_detector import M17TrackADetector
from src.models.m18_detector import M18ReenactmentDetector
from src.models.evidence_fusion import (
    ConstrainedLinearFusion,
    CalibratedLogOddsFusion,
    RankAverageFusion,
    HierarchicalGatedFusion,
    derive_threshold_policies,
)
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("calibrate_m22_evidence_fusion")

V5_CHAMPION_CKPT = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "best.pth"
V5_CALIB_CONFIG = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "calibration_config.json"
M17_TRACK_A_CKPT = PROJECT_ROOT / "checkpoints" / "m17_track_a_seed42" / "best.pth"
M18_DUAL_STREAM_CKPT = PROJECT_ROOT / "checkpoints" / "m18_dual_stream_seed42" / "best.pth"

MANIFEST_FULL = PROJECT_ROOT / "manifests" / "ffpp_c23.csv"

TENSOR_DIRS = [
    PROJECT_ROOT / "processed" / "m12_full",
    PROJECT_ROOT / "processed" / "m13_3_lofo",
]

OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "m22_ensemble"
CONFIG_OUT = OUTPUT_DIR / "fusion_config.json"
REPORT_JSON = PROJECT_ROOT / "reports" / "m22_fusion_calibration_report.json"
REPORT_MD = PROJECT_ROOT / "reports" / "m22_fusion_calibration_report.md"


def find_tensor(video_id: str, split: str = "val") -> Optional[Path]:
    """Find preprocessed tensor on disk."""
    candidates = [
        TENSOR_DIRS[0] / split / "tensors" / f"{video_id}.pt",
        TENSOR_DIRS[1] / split / "tensors" / f"{video_id}.pt",
        TENSOR_DIRS[0] / "test" / "tensors" / f"{video_id}.pt",
        TENSOR_DIRS[0] / "train" / "tensors" / f"{video_id}.pt",
        TENSOR_DIRS[1] / "test" / "tensors" / f"{video_id}.pt",
        TENSOR_DIRS[1] / "train" / "tensors" / f"{video_id}.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def main():
    logger.info("Initializing Milestone 22: Evidence Fusion & Calibration...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Frozen Models
    cfg = load_config("configs/experiments/m12_full.yaml")

    # M13.5 Champion
    base_v5 = DeepfakeDetector(config=cfg.model).to(device)
    v5_ckpt = torch.load(V5_CHAMPION_CKPT, map_location=device, weights_only=False)
    model_v5 = FamilyInvariantDeepfakeDetector(
        base_detector=base_v5, num_families=5, aux_hidden_dim=512, aux_dropout=0.5,
        freq_dim=64, texture_dim=64, fusion_dim=512, enable_mid_head=True,
    ).to(device)
    model_v5.load_state_dict(v5_ckpt["model_state_dict"])
    model_v5.eval()
    with open(V5_CALIB_CONFIG, "r", encoding="utf-8") as f:
        v5_calib = json.load(f)
    v5_temp = float(v5_calib.get("temperature", 2.0502))

    # M18 Dual-Stream
    base_m17 = DeepfakeDetector(config=cfg.model).to(device)
    m17_ckpt = torch.load(M17_TRACK_A_CKPT, map_location=device, weights_only=False)
    model_m17 = M17TrackADetector(
        base_detector=base_m17, num_families=5, res_dim=128, aux_hidden_dim=512,
        aux_dropout=0.5, gate_bottleneck=128,
    ).to(device)
    model_m17.load_state_dict(m17_ckpt["model_state_dict"])
    model_m17.eval()

    m18_ckpt = torch.load(M18_DUAL_STREAM_CKPT, map_location=device, weights_only=False)
    model_m18 = M18ReenactmentDetector(
        trunk_detector=model_m17, kin_dim=256, aux_hidden_dim=512,
        gate_bottleneck=128, freeze_trunk=True,
    ).to(device)
    model_m18.load_state_dict(m18_ckpt["model_state_dict"])
    model_m18.eval()

    logger.info("Both frozen models loaded successfully.")

    # 2. Collect Validation Predictions (N=1,255)
    df = pd.read_csv(MANIFEST_FULL)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    logger.info("Evaluating on %d FF++ validation videos...", len(val_df))

    records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for idx, row in val_df.iterrows():
            vid = str(row["video_id"])
            lbl = int(row["label"])
            fam = str(row.get("manipulation_type", "original"))
            tp = find_tensor(vid, split="val")
            if tp is None:
                continue

            t = torch.load(tp, map_location=device, weights_only=False)
            if t.dim() == 4:
                t = t.unsqueeze(0)

            with torch.amp.autocast("cuda"):
                out_v5 = model_v5(t)
                z_v5 = float(out_v5.primary_logits.item())
                p_v5 = float(torch.sigmoid(out_v5.primary_logits / v5_temp).item())

                out_18 = model_m18(t, mode="dual_stream")
                z_18 = float(out_18.primary_logits.item())
                p_18 = float(out_18.p_fused.item())

            records.append({
                "video_id": vid,
                "label": lbl,
                "family": fam,
                "z_13_5": z_v5,
                "p_13_5": p_v5,
                "z_18": z_18,
                "p_18": p_18,
            })

            if (idx + 1) % 250 == 0:
                logger.info("Processed %d/%d validation clips...", idx + 1, len(val_df))

    val_data = pd.DataFrame(records)
    logger.info("Successfully extracted validation predictions for %d clips (Reals: %d, Fakes: %d)",
                len(val_data), (val_data["label"] == 0).sum(), (val_data["label"] == 1).sum())

    labels = val_data["label"].values.astype(int)
    logits_matrix = np.column_stack([val_data["z_13_5"].values, val_data["z_18"].values])
    probas_matrix = np.column_stack([val_data["p_13_5"].values, val_data["p_18"].values])

    # Standalone Baselines on Val
    auc_13_5 = roc_auc_score(labels, val_data["z_13_5"].values)
    auc_18 = roc_auc_score(labels, val_data["z_18"].values)
    logger.info("Validation Standalone AUC: M13.5 = %.4f | M18 = %.4f", auc_13_5, auc_18)

    # 3. Fit Strategy 1: Constrained Linear Fusion
    strat1 = ConstrainedLinearFusion(num_models=2)
    strat1.fit(logits_matrix, labels)
    z_strat1 = strat1.predict_logits(logits_matrix)
    p_strat1 = strat1.predict_proba(logits_matrix)
    auc_strat1 = roc_auc_score(labels, z_strat1)
    pol_strat1 = derive_threshold_policies(p_strat1, labels)
    logger.info("Strategy 1 (Linear Fusion): w1=%.4f, w2=%.4f, b=%.4f -> Val AUC=%.4f (tau_95=%.4f)",
                strat1.weights[0], strat1.weights[1], strat1.bias, auc_strat1, pol_strat1.tau_95)

    # 4. Fit Strategy 2: Calibrated Log-Odds Platt Fusion
    strat2 = CalibratedLogOddsFusion(num_models=2)
    strat2.fit(probas_matrix, labels)
    p_strat2 = strat2.predict_proba(probas_matrix)
    auc_strat2 = roc_auc_score(labels, p_strat2)
    pol_strat2 = derive_threshold_policies(p_strat2, labels)
    logger.info("Strategy 2 (Log-Odds Platt): w1=%.4f, w2=%.4f, b=%.4f -> Val AUC=%.4f (tau_95=%.4f)",
                strat2.weights[0], strat2.weights[1], strat2.bias, auc_strat2, pol_strat2.tau_95)

    # 5. Fit Strategy 3: Non-Parametric Rank-Average Fusion
    strat3 = RankAverageFusion(num_models=2)
    strat3.fit(logits_matrix)
    s_strat3 = strat3.predict_rank_score(logits_matrix)
    auc_strat3 = roc_auc_score(labels, s_strat3)
    pol_strat3 = derive_threshold_policies(s_strat3, labels)
    logger.info("Strategy 3 (Rank-Average): Val AUC=%.4f (tau_95=%.4f)",
                auc_strat3, pol_strat3.tau_95)

    # 6. Fit Strategy 4: Hierarchical Gated Fusion
    strat4 = HierarchicalGatedFusion(tau_high=0.95, tau_low=0.05, fallback_fusion=strat1)
    p_strat4 = strat4.predict_proba(probas_matrix, logits_matrix)
    auc_strat4 = roc_auc_score(labels, p_strat4)
    pol_strat4 = derive_threshold_policies(p_strat4, labels)
    logger.info("Strategy 4 (Gated Fusion): Val AUC=%.4f (tau_95=%.4f)",
                auc_strat4, pol_strat4.tau_95)

    # Select Primary Champion Strategy: Highest Val AUC
    strategies_summary = {
        "strategy_1_constrained_linear": {
            "name": "Constrained Linear Logit Fusion",
            "weights": strat1.weights.tolist(),
            "bias": float(strat1.bias),
            "val_auc": float(auc_strat1),
            "policies": pol_strat1.to_dict(),
        },
        "strategy_2_log_odds_platt": {
            "name": "Calibrated Log-Odds Platt Fusion",
            "weights": strat2.weights.tolist(),
            "bias": float(strat2.bias),
            "val_auc": float(auc_strat2),
            "policies": pol_strat2.to_dict(),
        },
        "strategy_3_rank_average": {
            "name": "Non-Parametric Rank Average Fusion",
            "val_auc": float(auc_strat3),
            "policies": pol_strat3.to_dict(),
        },
        "strategy_4_hierarchical_gated": {
            "name": "Hierarchical Gated Routing",
            "tau_high": 0.95,
            "tau_low": 0.05,
            "val_auc": float(auc_strat4),
            "policies": pol_strat4.to_dict(),
        },
    }

    best_strat_key = max(strategies_summary, key=lambda k: strategies_summary[k]["val_auc"])
    logger.info("Optimal Strategy Selected on Validation: %s (AUC=%.4f)",
                best_strat_key, strategies_summary[best_strat_key]["val_auc"])

    # Save Configuration
    config_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "milestone": "M22",
        "description": "Validation-Fitted Multi-Model Decision-Level Evidence Fusion Configuration",
        "models": {
            "model_1": "Champion M13.5 (FamilyInvariantDeepfakeDetector, T=2.0502)",
            "model_2": "Baseline M18 (M18ReenactmentDetector Dual-Stream)",
        },
        "validation_samples": len(val_data),
        "validation_standalone_auc": {
            "m13_5": float(auc_13_5),
            "m18": float(auc_18),
        },
        "primary_selected_strategy": best_strat_key,
        "strategies": strategies_summary,
    }

    with open(CONFIG_OUT, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
    logger.info("Saved M22 fusion configuration to: %s", CONFIG_OUT)

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    # Markdown Report
    md_lines = [
        "# Milestone 22 — Decision-Level Evidence Fusion Calibration Report",
        "",
        f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        "**Methodology**: Zero Neural Network Retraining (Frozen Models Fitted on Validation Split)  ",
        f"**Validation Dataset**: FF++ C23 Validation Split ($N = {len(val_data)}$ clips)  ",
        "",
        "---",
        "",
        "## 1. Standalone Validation Performance",
        "",
        f"- **Champion M13.5 Alone**: ROC-AUC = **{auc_13_5:.4f}**",
        f"- **Baseline M18 Alone**: ROC-AUC = **{auc_18:.4f}**",
        "",
        "---",
        "",
        "## 2. Evidence Fusion Strategies Benchmark on Validation",
        "",
        "| Strategy | Model 1 Weight (M13.5) | Model 2 Weight (M18) | Bias $b^*$ | Val ROC-AUC | Val $\\tau_{95}^*$ | Val $\\tau_{99}^*$ |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| **1. Constrained Linear Logit** | {strat1.weights[0]:.4f} | {strat1.weights[1]:.4f} | {strat1.bias:+.4f} | **{auc_strat1:.4f}** | {pol_strat1.tau_95:.4f} | {pol_strat1.tau_99:.4f} |",
        f"| **2. Calibrated Log-Odds Platt** | {strat2.weights[0]:.4f} | {strat2.weights[1]:.4f} | {strat2.bias:+.4f} | **{auc_strat2:.4f}** | {pol_strat2.tau_95:.4f} | {pol_strat2.tau_99:.4f} |",
        f"| **3. Non-Parametric Rank Average** | 0.5000 | 0.5000 | 0.0000 | **{auc_strat3:.4f}** | {pol_strat3.tau_95:.4f} | {pol_strat3.tau_99:.4f} |",
        f"| **4. Hierarchical Gated Routing** | Gated ($p_{{13}} \\ge 0.95$) | Gated ($p_{{18}} \\le 0.05$) | — | **{auc_strat4:.4f}** | {pol_strat4.tau_95:.4f} | {pol_strat4.tau_99:.4f} |",
        "",
        "---",
        "",
        f"## 3. Primary Selected Strategy: **{best_strat_key}**",
        f"- Achieved highest validation ROC-AUC (**{strategies_summary[best_strat_key]['val_auc']:.4f}**).",
        f"- Locked operating thresholds: $\\tau_{{95}}^* = {strategies_summary[best_strat_key]['policies']['tau_95']:.4f}$, $\\tau_{{99}}^* = {strategies_summary[best_strat_key]['policies']['tau_99']:.4f}$.",
        "- All fusion parameters locked for evaluation across the 7 pre-registered promotion gates.",
        "",
    ]

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info("Saved M22 calibration markdown report to: %s", REPORT_MD)
    logger.info("Milestone 22 Calibration Complete.")


if __name__ == "__main__":
    main()
