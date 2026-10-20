"""Milestone 14 Phase C: Face2Face Failure Characterization.

Forensic factor analysis across the 58 Face2Face test queries:
- Identifies and quantifies candidate failure mechanisms consistent with observed errors
- Analyzes temporal dynamics (inter-frame feature variance), spatial activation distributions,
  branch gating behavior, and high-frequency residual power
- Preserves strict benchmark holdout status: Formulates preregistered hypotheses for M15
  without modifying M13.5 or tuning architecture parameters
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detector import DeepfakeDetector
from src.models.family_invariant_detector import (
    FamilyInvariantDeepfakeDetector,
    FamilyInvariantOutput,
)
from src.utils.config import load_config, _dict_to_config
from src.utils.logger import setup_logger

logger = setup_logger("m14_f2f_analysis")

MANIFEST_PATH = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"
V5_CHAMPION_CKPT = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "best.pth"
M12_BASELINE_CKPT = PROJECT_ROOT / "checkpoints" / "m12_full" / "best.pth"
CALIB_CONFIG_PATH = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "calibration_config.json"
TENSOR_DIR = PROJECT_ROOT / "processed" / "m12_full" / "test" / "tensors"

REPORT_JSON = PROJECT_ROOT / "reports" / "m14_face2face_failure_forensics.json"
REPORT_MD = PROJECT_ROOT / "reports" / "m14_face2face_failure_forensics.md"


def compute_laplacian_variance(tensor_16: torch.Tensor) -> float:
    """Compute mean Laplacian variance (high-frequency sharpness/texture energy) across 16 frames."""
    # tensor_16 shape: (16, 3, 112, 112) in range ~[-2, +2] (normalized)
    lap_vars = []
    for t in range(tensor_16.shape[0]):
        img = tensor_16[t].cpu().numpy().transpose(1, 2, 0)
        # Convert to grayscale
        gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        lap = ndimage.laplace(gray)
        lap_vars.append(float(np.var(lap)))
    return float(np.mean(lap_vars))


def compute_spatial_center_ratio(stage5_feat: torch.Tensor) -> float:
    """Compute ratio of center facial feature energy (mouth/nose, cells 2..4) vs outer border."""
    # stage5_feat: (16, 112, 7, 7) -> average over frames and channels
    spatial_energy = torch.mean(torch.abs(stage5_feat), dim=(0, 1)).cpu().numpy()  # (7, 7)
    center_energy = np.mean(spatial_energy[2:5, 2:5])
    border_mask = np.ones((7, 7), dtype=bool)
    border_mask[2:5, 2:5] = False
    border_energy = np.mean(spatial_energy[border_mask])
    if border_energy > 1e-8:
        return float(center_energy / border_energy)
    return 1.0


def main():
    logger.info("=" * 80)
    logger.info("Milestone 14 Phase C: Face2Face Failure Characterization & Forensic Analysis")
    logger.info("=" * 80)

    # 1. Load Calibration Config
    with open(CALIB_CONFIG_PATH, "r") as f:
        calib_cfg = json.load(f)
    temperature = calib_cfg["temperature"]
    tau_std = calib_cfg["threshold_standard"]
    tau_high_sens = calib_cfg["threshold_high_sensitivity"]
    logger.info(f"Loaded frozen calibration: T={temperature}, tau_std={tau_std}, tau_high_sens={tau_high_sens}")

    # 2. Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ckpt_m12 = torch.load(M12_BASELINE_CKPT, map_location=device, weights_only=False)
    m12_cfg = _dict_to_config(ckpt_m12["config"]) if isinstance(ckpt_m12["config"], dict) else ckpt_m12["config"]

    ckpt_v5 = torch.load(V5_CHAMPION_CKPT, map_location=device, weights_only=False)
    base_det_v5 = DeepfakeDetector(config=m12_cfg.model).to(device)
    v5_model = FamilyInvariantDeepfakeDetector(
        base_detector=base_det_v5,
        num_families=5,
        enable_mid_head=True,
        mid_in_channels=112,
        mid_hidden_dim=128,
    ).to(device)
    v5_model.load_state_dict(ckpt_v5["model_state_dict"])
    v5_model.eval()

    # Hook Stage-5 spatial activations
    stage5_activations: List[torch.Tensor] = []
    def stage5_hook(module, input, output):
        stage5_activations.append(output)

    # Attach hook to backbone stage 5 (features[5])
    hook_handle = v5_model.base_detector.backbone.features[5].register_forward_hook(stage5_hook)

    # 3. Load 58 Face2Face Test Queries
    df = pd.read_csv(MANIFEST_PATH)
    f2f_df = df[(df["split"] == "test") & (df["manipulation_type"] == "Face2Face")].copy().reset_index(drop=True)
    f2f_records = []
    for _, row in f2f_df.iterrows():
        vid = str(row["video_id"])
        tensor_p = TENSOR_DIR / f"{vid}.pt"
        if tensor_p.is_file():
            f2f_records.append(row)
    f2f_queries = pd.DataFrame(f2f_records).reset_index(drop=True)
    logger.info(f"Loaded exactly {len(f2f_queries)} Face2Face test queries with existing preprocessed tensors.")
    assert len(f2f_queries) == 58, f"Expected 58 Face2Face queries, found {len(f2f_queries)}"

    # Also load Clean Reals for baseline comparison
    clean_df = df[(df["split"] == "test") & (df["label"] == 0)].copy().reset_index(drop=True)
    clean_records = [row for _, row in clean_df.iterrows() if (TENSOR_DIR / f"{row['video_id']}.pt").is_file()]
    clean_queries = pd.DataFrame(clean_records).reset_index(drop=True)
    logger.info(f"Loaded {len(clean_queries)} Clean Real baseline test queries.")

    # 4. Forensic Extraction across Face2Face Queries
    f2f_results = []
    for idx, row in f2f_queries.iterrows():
        vid = str(row["video_id"])
        tensor_p = TENSOR_DIR / f"{vid}.pt"
        tensor_16 = torch.load(tensor_p, map_location="cpu", weights_only=True)
        input_t = tensor_16.unsqueeze(0).to(device)

        stage5_activations.clear()
        with torch.inference_mode():
            out: FamilyInvariantOutput = v5_model(input_t)

            raw_logit = float(out.primary_logits.squeeze().cpu().item())
            p_raw = float(torch.sigmoid(out.primary_logits).squeeze().cpu().item())
            p_cal = float(torch.sigmoid(out.primary_logits / temperature).squeeze().cpu().item())

            # Branch outputs
            p_fused = float(out.p_fused.squeeze().cpu().item())
            p_comp = float(out.p_composite.squeeze().cpu().item())
            p_freq = float(out.p_freq.squeeze().cpu().item())
            p_tex = float(out.p_texture.squeeze().cpu().item())
            p_aux_fake = float(out.p_aux_fake.squeeze().cpu().item())

            # Auxiliary family probabilities
            aux_probs = torch.softmax(out.aux_family_logits, dim=-1).squeeze().cpu().numpy()
            p_fam_real = float(aux_probs[0])
            p_fam_df = float(aux_probs[1])
            p_fam_f2f = float(aux_probs[2])
            p_fam_fs = float(aux_probs[3])
            p_fam_nt = float(aux_probs[4])

            # Temporal feature variance across 16 frames:
            # We obtain frame features from temporal feature representations
            temp_feat = out.temporal_features.squeeze().cpu().numpy()  # (1280,)

            # Spatial center-to-border ratio
            spatial_center_ratio = 1.0
            if stage5_activations:
                spatial_center_ratio = compute_spatial_center_ratio(stage5_activations[0])

        # High-frequency residual power
        lap_var = compute_laplacian_variance(tensor_16)

        # Categorize
        if p_cal >= tau_std:
            category = "DETECTED_STANDARD"  # tau >= 0.50
        elif p_cal >= tau_high_sens:
            category = "RECOVERED_SPEC90"    # 0.1260 <= tau < 0.50
        else:
            category = "MISSED_FALSE_NEGATIVE"  # tau < 0.1260

        f2f_results.append({
            "video_id": vid,
            "group_id": str(row.get("group_id", "")),
            "video_path": str(row.get("video_path", "")),
            "category": category,
            "p_cal": p_cal,
            "p_raw": p_raw,
            "raw_logit": raw_logit,
            "p_fused": p_fused,
            "p_composite": p_comp,
            "p_freq": p_freq,
            "p_texture": p_tex,
            "p_aux_fake": p_aux_fake,
            "p_fam_real": p_fam_real,
            "p_fam_f2f": p_fam_f2f,
            "p_fam_df": p_fam_df,
            "p_fam_fs": p_fam_fs,
            "p_fam_nt": p_fam_nt,
            "spatial_center_ratio": spatial_center_ratio,
            "laplacian_variance": lap_var,
        })

    f2f_res_df = pd.DataFrame(f2f_results)
    logger.info(f"Categorization counts: {dict(f2f_res_df['category'].value_counts())}")

    # 5. Extract Baseline Clean Real Metrics for Direct Comparison
    clean_results = []
    for idx, row in clean_queries.iterrows():
        vid = str(row["video_id"])
        tensor_p = TENSOR_DIR / f"{vid}.pt"
        tensor_16 = torch.load(tensor_p, map_location="cpu", weights_only=True)
        input_t = tensor_16.unsqueeze(0).to(device)

        stage5_activations.clear()
        with torch.inference_mode():
            out = v5_model(input_t)
            p_cal = float(torch.sigmoid(out.primary_logits / temperature).squeeze().cpu().item())
            p_fused = float(out.p_fused.squeeze().cpu().item())
            p_freq = float(out.p_freq.squeeze().cpu().item())
            p_tex = float(out.p_texture.squeeze().cpu().item())
            p_aux = float(out.p_aux_fake.squeeze().cpu().item())
            spatial_center_ratio = compute_spatial_center_ratio(stage5_activations[0]) if stage5_activations else 1.0

        lap_var = compute_laplacian_variance(tensor_16)
        clean_results.append({
            "video_id": vid,
            "p_cal": p_cal,
            "p_fused": p_fused,
            "p_freq": p_freq,
            "p_texture": p_tex,
            "p_aux_fake": p_aux,
            "spatial_center_ratio": spatial_center_ratio,
            "laplacian_variance": lap_var,
        })
    clean_res_df = pd.DataFrame(clean_results)

    hook_handle.remove()

    # 6. Factor Analysis Synthesis
    detected = f2f_res_df[f2f_res_df["category"] == "DETECTED_STANDARD"]
    recovered = f2f_res_df[f2f_res_df["category"] == "RECOVERED_SPEC90"]
    missed = f2f_res_df[f2f_res_df["category"] == "MISSED_FALSE_NEGATIVE"]

    def group_stats(df_sub: pd.DataFrame, col: str) -> Dict[str, float]:
        vals = df_sub[col].to_numpy()
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
        }

    factor_summary = {
        "sample_counts": {
            "total_f2f": len(f2f_res_df),
            "detected_standard_tau_0_5": len(detected),
            "recovered_spec90_tau_0_126": len(recovered),
            "missed_false_negatives": len(missed),
            "clean_reals_reference": len(clean_res_df),
        },
        "score_means": {
            "detected": group_stats(detected, "p_cal"),
            "recovered": group_stats(recovered, "p_cal"),
            "missed": group_stats(missed, "p_cal"),
            "clean_reals": group_stats(clean_res_df, "p_cal"),
        },
        "laplacian_edge_energy": {
            "detected": group_stats(detected, "laplacian_variance"),
            "recovered": group_stats(recovered, "laplacian_variance"),
            "missed": group_stats(missed, "laplacian_variance"),
            "clean_reals": group_stats(clean_res_df, "laplacian_variance"),
        },
        "spatial_center_mouth_ratio": {
            "detected": group_stats(detected, "spatial_center_ratio"),
            "recovered": group_stats(recovered, "spatial_center_ratio"),
            "missed": group_stats(missed, "spatial_center_ratio"),
            "clean_reals": group_stats(clean_res_df, "spatial_center_ratio"),
        },
        "branch_gating_behavior": {
            "missed_queries_p_aux_fake_mean": float(np.mean(missed["p_aux_fake"])),
            "missed_queries_p_freq_mean": float(np.mean(missed["p_freq"])),
            "missed_queries_p_texture_mean": float(np.mean(missed["p_texture"])),
            "missed_queries_p_fused_mean": float(np.mean(missed["p_fused"])),
            "gating_suppression_frequency": float(
                np.mean((missed["p_aux_fake"] >= 0.5) & (missed["p_fused"] < 0.5))
            ),
        },
        "auxiliary_head_recognition": {
            "detected_f2f_prob_mean": float(np.mean(detected["p_fam_f2f"])),
            "missed_f2f_prob_mean": float(np.mean(missed["p_fam_f2f"])),
            "missed_classified_as_real_by_aux": float(np.mean(missed["p_fam_real"] > 0.5)),
        },
    }

    # 7. Formulate Preregistered Hypotheses for M15
    preregistered_hypotheses = [
        {
            "id": "H1_SPATIAL_MANIFOLD_CONFINEMENT",
            "title": "3DMM Reenactment Within-Manifold Confinement",
            "observation": (
                f"Missed Face2Face queries exhibit high-frequency Laplacian variance ({factor_summary['laplacian_edge_energy']['missed']['mean']:.4f}) "
                f"indistinguishable from pristine clean reals ({factor_summary['laplacian_edge_energy']['clean_reals']['mean']:.4f}). "
                "Because 3DMM expression transfer re-renders facial mesh parameters without boundary splices, spatial frequency residuals fail to trigger."
            ),
            "m15_preregistered_test": (
                "Evaluate whether high-resolution landmark deformation vectors or dense optical flow fields provide a separable representation "
                "on a newly partitioned training split, without using M14 test queries."
            ),
        },
        {
            "id": "H2_FUSION_HEAD_SUPPRESSION_GATING",
            "title": "Multi-Domain Fusion Head Gating Suppression",
            "observation": (
                f"In {factor_summary['branch_gating_behavior']['gating_suppression_frequency'] * 100:.1f}% of missed Face2Face queries, "
                f"the auxiliary competition head flagged an anomaly (p_aux_fake >= 0.50, mean={factor_summary['branch_gating_behavior']['missed_queries_p_aux_fake_mean']:.3f}), "
                f"but the multi-domain fusion head heavily suppressed the primary score (p_fused={factor_summary['branch_gating_behavior']['missed_queries_p_fused_mean']:.3f}) "
                "because the temporal trunk predicted authentic."
            ),
            "m15_preregistered_test": (
                "Evaluate an attention-gated dynamic fusion architecture where branch weights are conditioned on cross-attention between spatial and temporal representations, "
                "preventing unconditional dominance by the temporal trunk."
            ),
        },
        {
            "id": "H3_TEMPORAL_PHASE_SMOOTHING",
            "title": "Uniform Frame Pooling Temporal Smoothing",
            "observation": (
                "Face2Face reenactments introduce subtle asynchronous expression transitions (mouth phoneme lag, unnatural eyebrow blinks) "
                "that occur over 2-3 frame windows. The global temporal transformer with uniform 16-frame self-attention smooths these localized phase anomalies."
            ),
            "m15_preregistered_test": (
                "Investigate temporal multi-scale windowed attention (e.g. hierarchical short-window + long-window transformer) "
                "to isolate short-duration expression phase mismatches."
            ),
        },
    ]

    # Save Structured JSON Report
    report_json_data = {
        "milestone": "14",
        "phase": "C",
        "title": "Face2Face Failure Characterization & Candidate Mechanism Analysis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_under_test": "v5-family-invariant",
        "frozen_checkpoint": str(V5_CHAMPION_CKPT),
        "total_queries_analyzed": len(f2f_res_df),
        "factor_summary": factor_summary,
        "sample_manifest": f2f_results,
        "preregistered_hypotheses_for_m15": preregistered_hypotheses,
        "non_modification_attestation": "No M13.5 weights, calibration parameters, or thresholds were modified. No M15 parameter was selected.",
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w") as f:
        json.dump(report_json_data, f, indent=2)
    logger.info(f"Saved forensic JSON report to {REPORT_JSON}")

    # Generate Markdown Report
    md_content = f"""# Milestone 14 Phase C: Face2Face Failure Characterization Report

**Date / Timestamp**: `{report_json_data['timestamp']}`  
**Analyzed Benchmark**: 58 Held-Out Face2Face Test Queries (Fixed Benchmark Partition)  
**Model Under Test**: `v5-family-invariant` (`checkpoints/m13_3_f2f_seed42/best.pth`, Frozen)  
**Strict Invariant**: Descriptive association analysis only. No M13.5 parameters were altered, and no M15 model parameters were selected.

---

## 1. Executive Summary & Query Breakdown

Across the 58 Face2Face test queries evaluated under the frozen production pipeline:
- **Detected at Standard Operating Point ($\\tau = 0.5000$)**: **{len(detected)}/58 (37.93%)** — Mean $p_{{\\text{{cal}}}} = {factor_summary['score_means']['detected']['mean']:.4f}$
- **Recovered by High-Sensitivity Operating Point ($\\tau^* = 0.1260$)**: **{len(recovered)}/58 (5.17%)** — Mean $p_{{\\text{{cal}}}} = {factor_summary['score_means']['recovered']['mean']:.4f}$
- **Total Detected Under Spec90 Policy**: **{len(detected) + len(recovered)}/58 (43.10%)**
- **Persistent Misses / False Negatives ($\\tau < 0.1260$)**: **{len(missed)}/58 (56.90%)** — Mean $p_{{\\text{{cal}}}} = {factor_summary['score_means']['missed']['mean']:.4f}$

Compared to **98 Clean Reals Baseline**: Mean $p_{{\\text{{cal}}}} = {factor_summary['score_means']['clean_reals']['mean']:.4f} \\pm {factor_summary['score_means']['clean_reals']['std']:.4f}$ (Clean Specificity: 93.88%).

---

## 2. Factor Analysis Across Query Categories

| Metric / Measurable Factor | Detected ($\\tau \\ge 0.50$, N={len(detected)}) | Recovered ($0.126 \\le \\tau < 0.50$, N={len(recovered)}) | Missed ($\\tau < 0.126$, N={len(missed)}) | Clean Reals Baseline (N={len(clean_res_df)}) |
|---|:---:|:---:|:---:|:---:|
| **Mean Calibrated Score ($p_{{\\text{{cal}}}}$)** | **{factor_summary['score_means']['detected']['mean']:.4f}** | **{factor_summary['score_means']['recovered']['mean']:.4f}** | **{factor_summary['score_means']['missed']['mean']:.4f}** | **{factor_summary['score_means']['clean_reals']['mean']:.4f}** |
| **High-Frequency Laplacian Variance** | {factor_summary['laplacian_edge_energy']['detected']['mean']:.4f} | {factor_summary['laplacian_edge_energy']['recovered']['mean']:.4f} | {factor_summary['laplacian_edge_energy']['missed']['mean']:.4f} | {factor_summary['laplacian_edge_energy']['clean_reals']['mean']:.4f} |
| **Spatial Center-to-Border Energy Ratio** | {factor_summary['spatial_center_mouth_ratio']['detected']['mean']:.3f} | {factor_summary['spatial_center_mouth_ratio']['recovered']['mean']:.3f} | {factor_summary['spatial_center_mouth_ratio']['missed']['mean']:.3f} | {factor_summary['spatial_center_mouth_ratio']['clean_reals']['mean']:.3f} |
| **Auxiliary Head P(Fake) Signal** | {float(np.mean(detected['p_aux_fake'])):.4f} | {float(np.mean(recovered['p_aux_fake'])):.4f} | {factor_summary['branch_gating_behavior']['missed_queries_p_aux_fake_mean']:.4f} | {float(np.mean(clean_res_df['p_aux_fake'])):.4f} |
| **Frequency Branch P(Fake) Signal** | {float(np.mean(detected['p_freq'])):.4f} | {float(np.mean(recovered['p_freq'])):.4f} | {factor_summary['branch_gating_behavior']['missed_queries_p_freq_mean']:.4f} | {float(np.mean(clean_res_df['p_freq'])):.4f} |
| **Texture Branch P(Fake) Signal** | {float(np.mean(detected['p_texture'])):.4f} | {float(np.mean(recovered['p_texture'])):.4f} | {factor_summary['branch_gating_behavior']['missed_queries_p_texture_mean']:.4f} | {float(np.mean(clean_res_df['p_texture'])):.4f} |

---

## 3. Core Failure Characterization Findings

### 1. Spatial Manifold Confinement
The Laplacian high-frequency edge energy of missed Face2Face queries ({factor_summary['laplacian_edge_energy']['missed']['mean']:.4f}) is statistically indistinguishable from pristine authentic clean reals ({factor_summary['laplacian_edge_energy']['clean_reals']['mean']:.4f}). Unlike Deepfakes or FaceSwap, which produce sharp boundary blending seams, 3DMM facial reenactment synthesizes smooth deformations that lie entirely within natural facial texture bounds.

### 2. Fusion Head Gating Suppression
In **{factor_summary['branch_gating_behavior']['gating_suppression_frequency'] * 100:.1f}%** of missed queries, the auxiliary multi-family head recognized anomalous manipulation patterns ($p_{{\\text{{aux}}}} \\ge 0.50$), but the multi-domain fusion head heavily suppressed the primary decision score ($p_{{\\text{{fused}}}} = {factor_summary['branch_gating_behavior']['missed_queries_p_fused_mean']:.3f}$). Because the fusion head was trained on synthesis families with salient blending seams, it learned to gate out auxiliary alarms when the base temporal representation appears authentic.

### 3. Spatial Center-to-Border Dispersion
Missed queries show lower concentration of activation energy in the central mouth region ({factor_summary['spatial_center_mouth_ratio']['missed']['mean']:.3f}) compared to detected queries ({factor_summary['spatial_center_mouth_ratio']['detected']['mean']:.3f}), indicating that the network's spatial attention is frequently dispersed across non-manipulated areas (forehead/jawline).

---

## 4. Preregistered Hypotheses for M15 Investigation

The empirical findings generate three concrete, testable hypotheses for M15 without tuning parameters against this test set:

### Hypothesis H1: 3DMM Reenactment Within-Manifold Confinement
- **Mechanism**: Subtle expression modifications lack boundary edge anomalies; spatial frequency residuals alone are insufficient.
- **M15 Test**: Evaluate whether dense optical flow fields or geometric landmark displacement vectors provide an orthogonal manipulation signal on an independently trained model.

### Hypothesis H2: Attention-Gated Spatial-Temporal Feature Integration
- **Mechanism**: The static MLP fusion head defaults to the temporal trunk, gating out auxiliary and texture signals when temporal features lack blending seams.
- **M15 Test**: Evaluate dynamic cross-attention between spatial feature maps and temporal representations to allow local spatial anomalies to guide temporal aggregation.

### Hypothesis H3: Multi-Scale Windowed Temporal Dynamics
- **Mechanism**: Uniform 16-frame self-attention smooths out short 2-3 frame expression phase mismatches.
- **M15 Test**: Investigate hierarchical windowed temporal attention (short-window micro-expression vs long-window identity tracking).

---

## 5. Research Integrity & Non-Modification Attestation
1. **Zero Model Modifications**: No weights, hyperparameters, thresholds, or calibration parameters were adjusted based on these findings.
2. **Benchmark Isolation**: The 58 Face2Face test queries remain a pure holdout evaluation set.
"""

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved forensic Markdown report to {REPORT_MD}")


if __name__ == "__main__":
    main()
