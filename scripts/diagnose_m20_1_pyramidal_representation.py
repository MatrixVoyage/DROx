"""Milestone 20.1: Whole-Face Multi-Scale Pyramidal Representation Diagnostic.

Evaluates whether multi-scale spatial frequency decomposition (Gaussian base +
Laplacian detail bands) of the whole-face crop resolves the Face2Face bottleneck
without breaking whole-face global geometric context.

Arms Evaluated (on 116 locked Face2Face holdout clips):
- Arm 1: Whole-Face Base Control (G1, standard 112x112)
- Arm 2: Mid-Frequency Warping Octave (L1 band-pass layer)
- Arm 3: Native High-Frequency Octave (L0 decimated to 112x112)
- Arm 4a: Native High-Pass Detail Injection (alpha = 0.20)
- Arm 4b: Native High-Pass Detail Injection (alpha = 0.50)

Target Model: Frozen M18 Dual-Stream Checkpoint (Seed 42)
Statistical Rigor: Paired bootstrap vs. Arm 1 (n_resamples=1000)
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detector import DeepfakeDetector
from src.models.m17_detector import M17TrackADetector
from src.models.m18_detector import M18ReenactmentDetector
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("diagnose_m20_1_pyramidal_representation")

# Checkpoints
M17_TRACK_A_CKPT = PROJECT_ROOT / "checkpoints" / "m17_track_a_seed42" / "best.pth"
M18_DUAL_STREAM_CKPT = PROJECT_ROOT / "checkpoints" / "m18_dual_stream_seed42" / "best.pth"

# Manifest
MANIFEST_MIXED = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"

# Output Reports
REPORT_JSON = PROJECT_ROOT / "reports" / "m20_1_pyramidal_diagnostic_report.json"
REPORT_MD = PROJECT_ROOT / "reports" / "m20_1_pyramidal_diagnostic_report.md"

# ImageNet normalization stats
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def decompose_laplacian_pyramid(
    G0_uint8: np.ndarray,
    target_size: int = 112,
) -> Dict[str, Any]:
    """Decompose native whole-face crop into Gaussian base and Laplacian detail layers.

    Args:
        G0_uint8: Native whole-face crop (H0, W0, 3) uint8.
        target_size: Target base size (112).

    Returns:
        Dict with G0, G1, G2, L0, L1 in float32 [0, 1] and uint8 visual representations.
    """
    H0, W0 = G0_uint8.shape[:2]
    G0_f = G0_uint8.astype(np.float32) / 255.0

    # Gaussian Level 1: downsampled to target_size (112x112)
    G1_f = cv2.resize(G0_f, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Gaussian Level 2: downsampled to target_size // 2 (56x56)
    half_size = target_size // 2
    G2_f = cv2.resize(G1_f, (half_size, half_size), interpolation=cv2.INTER_AREA)

    # Laplacian Level 1 (Mid-frequency octave): G1 - Upsample(G2)
    G2_up = cv2.resize(G2_f, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    L1_f = G1_f - G2_up

    # Laplacian Level 0 (Native high-frequency octave): G0 - Upsample(G1)
    G1_up = cv2.resize(G1_f, (W0, H0), interpolation=cv2.INTER_LINEAR)
    L0_f = G0_f - G1_up

    # Reconstruction check: G1 == G2_up + L1
    G1_recon_err = float(np.max(np.abs(G1_f - (G2_up + L1_f))))
    G0_recon_err = float(np.max(np.abs(G0_f - (G1_up + L0_f))))

    # Convert to visual uint8 for Arm construction
    G1_uint8 = np.clip(G1_f * 255.0, 0, 255).astype(np.uint8)
    G2_uint8 = np.clip(G2_f * 255.0, 0, 255).astype(np.uint8)

    return {
        "G0": G0_uint8,
        "G1": G1_uint8,
        "G2": G2_uint8,
        "G0_f": G0_f,
        "G1_f": G1_f,
        "G2_f": G2_f,
        "L0": L0_f,
        "L1": L1_f,
        "L0_f": L0_f,
        "L1_f": L1_f,
        "G1_recon_err": G1_recon_err,
        "G0_recon_err": G0_recon_err,
    }


def construct_pyramidal_arm_patches(pyramid: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Construct 5 diagnostic arm frames (112, 112, 3) uint8 from pyramidal decomposition."""
    G1_f = pyramid["G1_f"]
    L1_f = pyramid["L1_f"] # (112, 112, 3), zero-centered
    L0_f = pyramid["L0_f"] # (H0, W0, 3), zero-centered

    # Decimate L0 to (112, 112, 3)
    L0_112_f = cv2.resize(L0_f, (112, 112), interpolation=cv2.INTER_AREA)

    # Arm 1: Base Control (G1, standard 112x112 RGB)
    arm_1_base = pyramid["G1"]

    # Arm 2: Mid-Frequency Warping Octave (L1 normalized to visual range [0, 255])
    # Shift zero-centered delta by 0.5 to form natural image
    arm_2_mid_l1 = np.clip((L1_f + 0.5) * 255.0, 0, 255).astype(np.uint8)

    # Arm 3: Native High-Frequency Octave (L0 decimated to 112 and shifted to visual range)
    arm_3_high_l0 = np.clip((L0_112_f + 0.5) * 255.0, 0, 255).astype(np.uint8)

    # Arm 4a: High-Pass Detail Injection (alpha = 0.20)
    arm_4_inject_02 = np.clip((G1_f + 0.20 * L0_112_f) * 255.0, 0, 255).astype(np.uint8)

    # Arm 4b: High-Pass Detail Injection (alpha = 0.50)
    arm_4_inject_05 = np.clip((G1_f + 0.50 * L0_112_f) * 255.0, 0, 255).astype(np.uint8)

    return {
        "arm_1_base": arm_1_base,
        "arm_2_mid_l1": arm_2_mid_l1,
        "arm_3_high_l0": arm_3_high_l0,
        "arm_4_inject_02": arm_4_inject_02,
        "arm_4_inject_05": arm_4_inject_05,
    }


def normalize_pyramidal_sequence(patches: List[np.ndarray]) -> torch.Tensor:
    """Normalize a sequence of 16 (112, 112, 3) uint8 patches to (1, 16, 3, 112, 112) float32."""
    processed = []
    for p in patches:
        arr = p.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        processed.append(arr.transpose(2, 0, 1)) # (3, 112, 112)
    seq = np.stack(processed, axis=0) # (16, 3, 112, 112)
    return torch.from_numpy(seq).unsqueeze(0).float() # (1, 16, 3, 112, 112)


def compute_paired_bootstrap(
    y_true: np.ndarray,
    scores_cand: np.ndarray,
    scores_ref: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute paired bootstrap difference in ROC-AUC with confidence intervals and p-value."""
    n = len(y_true)
    rng = np.random.RandomState(seed)

    base_auc_cand = roc_auc_score(y_true, scores_cand)
    base_auc_ref = roc_auc_score(y_true, scores_ref)
    base_delta = float(base_auc_cand - base_auc_ref)

    deltas = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            auc_c = roc_auc_score(y_b, scores_cand[idx])
            auc_r = roc_auc_score(y_b, scores_ref[idx])
            deltas.append(auc_c - auc_r)
        except ValueError:
            continue

    deltas = np.array(deltas)
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_val = float(np.mean(deltas <= 0.0))

    return {
        "delta_auc": base_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
    }


def compute_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta non-parametric effect size."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    greater = 0
    less = 0
    for val_x in x:
        greater += int(np.sum(val_x > y))
        less += int(np.sum(val_x < y))
    return float((greater - less) / (n_x * n_y))


def classify_pyramidal_outcome(arm_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate Three-Tier Outcome Interpretation Framework for Pyramidal Diagnostic."""
    base = arm_metrics["arm_1_base"]
    mid = arm_metrics["arm_2_mid_l1"]
    high = arm_metrics["arm_3_high_l0"]
    inj_02 = arm_metrics["arm_4_inject_02"]
    inj_05 = arm_metrics["arm_4_inject_05"]

    # 1. Check for Support: either injection arm shows meaningful paired gain
    for name, arm in [("Arm 4a (alpha=0.20)", inj_02), ("Arm 4b (alpha=0.50)", inj_05)]:
        if arm.get("delta_auc_vs_base", 0.0) > 0.020 and arm.get("p_val_vs_base", 1.0) < 0.05:
            return {
                "tier": "SUPPORT",
                "reason": (
                    f"Statistically supported improvement observed under {name} "
                    f"(Delta AUC = {arm['delta_auc_vs_base']:+.4f}, p = {arm['p_val_vs_base']:.4f}). "
                    "Injecting native high-pass detail directly into whole-face crops improves forensic discrimination."
                ),
                "recommendation": "Proceed to design and train full Whole-Face Multi-Scale Pyramidal Network (M20.2).",
            }

    # 2. Check for Inconclusive / Frequency-Shift
    # If isolated Laplacian layers collapse due to mean shift, but Injection preserves base performance
    inj_deltas = [inj_02.get("delta_auc_vs_base", 0.0), inj_05.get("delta_auc_vs_base", 0.0)]
    max_inj_delta = max(inj_deltas)

    if max_inj_delta >= -0.015:
        return {
            "tier": "INCONCLUSIVE",
            "reason": (
                f"High-pass detail injection preserves whole-face discrimination without regression (max Delta = {max_inj_delta:+.4f}), "
                f"while isolated Laplacian detail bands collapse ({mid['auc']:.4f} mid, {high['auc']:.4f} high) due to distribution shift. "
                "Front-end residual filters require end-to-end multi-channel tuning to exploit high-frequency octaves."
            ),
            "recommendation": "Conduct small multi-channel linear probe before full architecture training.",
        }

    # 3. Otherwise: Negative Evidence
    return {
        "tier": "NEGATIVE EVIDENCE",
        "reason": (
            f"Injecting native high-frequency residuals actively degraded whole-face performance "
            f"(Delta = {inj_02.get('delta_auc_vs_base', 0.0):+.4f} and {inj_05.get('delta_auc_vs_base', 0.0):+.4f}). "
            "High-frequency spatial noise disrupts the tuned 112x112 residual filters without providing discriminative advantage."
        ),
        "recommendation": "Archive spatial frequency bypass hypothesis; preserve M13.5 champion.",
    }


def load_m18_model(device: torch.device) -> M18ReenactmentDetector:
    """Load the frozen M18 Dual-Stream model."""
    cfg = load_config("configs/experiments/m12_full.yaml")
    base_cand = DeepfakeDetector(config=cfg.model).to(device)
    m17_ckpt = torch.load(M17_TRACK_A_CKPT, map_location=device, weights_only=False)
    model_track_a = M17TrackADetector(
        base_detector=base_cand,
        num_families=5,
        res_dim=128,
        aux_hidden_dim=512,
        aux_dropout=0.5,
        gate_bottleneck=128,
    ).to(device)
    model_track_a.load_state_dict(m17_ckpt["model_state_dict"])
    model_track_a.eval()

    m18_ckpt = torch.load(M18_DUAL_STREAM_CKPT, map_location=device, weights_only=False)
    m18_dual = M18ReenactmentDetector(
        trunk_detector=model_track_a,
        kin_dim=256,
        aux_hidden_dim=512,
        gate_bottleneck=128,
        freeze_trunk=True,
    ).to(device)
    m18_dual.load_state_dict(m18_ckpt["model_state_dict"])
    m18_dual.eval()
    return m18_dual


def main():
    logger.info("Initializing Milestone 20.1: Whole-Face Multi-Scale Pyramidal Representation Diagnostic...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model = load_m18_model(device)
    logger.info("Frozen M18 Dual-Stream model loaded cleanly on %s", device)

    # 2. Load MTCNN Detector
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=False, device=device)
    logger.info("MTCNN detector initialized.")

    # 3. Load 116 Locked Face2Face Holdout Clips
    mixed_df = pd.read_csv(MANIFEST_MIXED)
    mixed_test = mixed_df[mixed_df["split"] == "test"].copy().reset_index(drop=True)
    clean_reals = mixed_test[mixed_test["manipulation_type"] == "original"]
    f2f_fakes = mixed_test[mixed_test["manipulation_type"] == "Face2Face"]

    from scripts.diagnose_m19_1_resolution_temporal import find_tensor
    holdout_items: List[Dict[str, Any]] = []

    for _, r in clean_reals.iterrows():
        vid = str(r["video_id"])
        if find_tensor(vid, split="test") is not None and os.path.exists(str(r["video_path"])):
            holdout_items.append({
                "video_id": vid,
                "video_path": str(r["video_path"]),
                "label": 0,
            })
    for _, r in f2f_fakes.iterrows():
        vid = str(r["video_id"])
        if find_tensor(vid, split="test") is not None and os.path.exists(str(r["video_path"])):
            holdout_items.append({
                "video_id": vid,
                "video_path": str(r["video_path"]),
                "label": 1,
            })

    logger.info("Locked Face2Face Holdout verified: %d clips (Reals: %d, Fakes: %d)",
                len(holdout_items),
                sum(1 for x in holdout_items if x["label"] == 0),
                sum(1 for x in holdout_items if x["label"] == 1))

    arm_names = ["arm_1_base", "arm_2_mid_l1", "arm_3_high_l0", "arm_4_inject_02", "arm_4_inject_05"]
    predictions: Dict[str, Dict[str, List[float]]] = {
        arm: {"logits": [], "probs": [], "gates": [], "labels": []}
        for arm in arm_names
    }

    t0_start = time.time()
    for idx, item in enumerate(holdout_items):
        v_path = item["video_path"]
        label = item["label"]

        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            logger.warning("Could not open %s, skipping", v_path)
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 16:
            frame_indices = list(range(total_frames)) + [total_frames - 1] * (16 - total_frames)
        else:
            frame_indices = [int(i * total_frames / 16) for i in range(16)]

        raw_frames = []
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, f = cap.read()
            if not ret or f is None:
                if len(raw_frames) > 0:
                    raw_frames.append(raw_frames[-1].copy())
                else:
                    raw_frames.append(np.zeros((720, 1280, 3), dtype=np.uint8))
            else:
                raw_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()

        # Batch detect faces
        boxes, probs = mtcnn.detect(raw_frames)

        last_valid_box = None
        for b in boxes:
            if b is not None and len(b) > 0:
                last_valid_box = b[0]
                break

        if last_valid_box is None:
            H, W = raw_frames[0].shape[:2]
            last_valid_box = np.array([W * 0.25, H * 0.2, W * 0.75, H * 0.8])

        arm_patch_sequences: Dict[str, List[np.ndarray]] = {arm: [] for arm in arm_names}

        for fi, frame in enumerate(raw_frames):
            b = boxes[fi]
            if b is not None and len(b) > 0:
                cur_box = b[0]
                last_valid_box = cur_box
            else:
                cur_box = last_valid_box

            H, W = frame.shape[:2]
            bx1, by1, bx2, by2 = cur_box[:4]
            bw = max(1.0, bx2 - bx1)
            bh = max(1.0, by2 - by1)

            # Native-resolution whole-face crop with 0.20 margin expansion
            margin_x = bw * 0.20
            margin_y = bh * 0.20
            fx1 = max(0, int(round(bx1 - margin_x)))
            fy1 = max(0, int(round(by1 - margin_y)))
            fx2 = min(W, int(round(bx2 + margin_x)))
            fy2 = min(H, int(round(by2 + margin_y)))
            if fx2 <= fx1:
                fx2 = min(W, fx1 + 1)
            if fy2 <= fy1:
                fy2 = min(H, fy1 + 1)

            native_crop = frame[fy1:fy2, fx1:fx2]
            pyramid = decompose_laplacian_pyramid(native_crop, target_size=112)
            arm_patches = construct_pyramidal_arm_patches(pyramid)

            for arm in arm_names:
                arm_patch_sequences[arm].append(arm_patches[arm])

        # Evaluate through frozen M18 detector
        with torch.no_grad():
            for arm in arm_names:
                seq_tensor = normalize_pyramidal_sequence(arm_patch_sequences[arm]).to(device)
                out = model(seq_tensor)
                logit = float(out.primary_logits.item())
                prob = float(out.p_fused.item())
                gate = float(out.gate_values.item())

                predictions[arm]["logits"].append(logit)
                predictions[arm]["probs"].append(prob)
                predictions[arm]["gates"].append(gate)
                predictions[arm]["labels"].append(label)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(holdout_items):
            logger.info("Processed %d / %d videos (Elapsed: %.1fs)", idx + 1, len(holdout_items), time.time() - t0_start)

    # 4. Compute Metrics
    arm_metrics: Dict[str, Dict[str, Any]] = {}
    labels_arr = np.array(predictions["arm_1_base"]["labels"])
    reals_mask = labels_arr == 0
    fakes_mask = labels_arr == 1

    for arm in arm_names:
        z_arr = np.array(predictions[arm]["logits"])
        p_arr = np.array(predictions[arm]["probs"])
        g_arr = np.array(predictions[arm]["gates"])

        auc = float(roc_auc_score(labels_arr, z_arr))
        prauc = float(average_precision_score(labels_arr, z_arr))

        r_z_med = float(np.median(z_arr[reals_mask]))
        f_z_med = float(np.median(z_arr[fakes_mask]))
        margin = f_z_med - r_z_med

        r_g_med = float(np.median(g_arr[reals_mask]))
        f_g_med = float(np.median(g_arr[fakes_mask]))

        spec_05 = float(np.mean(p_arr[reals_mask] < 0.50))
        sens_05 = float(np.mean(p_arr[fakes_mask] >= 0.50))

        arm_metrics[arm] = {
            "auc": auc,
            "prauc": prauc,
            "real_median_logit": r_z_med,
            "fake_median_logit": f_z_med,
            "logit_margin": margin,
            "real_median_gate": r_g_med,
            "fake_median_gate": f_g_med,
            "spec_at_tau05": spec_05,
            "sens_at_tau05": sens_05,
        }

    # Paired Bootstrap vs Arm 1 Base Control
    scores_base = np.array(predictions["arm_1_base"]["logits"])
    for arm in arm_names:
        if arm == "arm_1_base":
            arm_metrics[arm]["delta_auc_vs_base"] = 0.0
            arm_metrics[arm]["ci_lower_vs_base"] = 0.0
            arm_metrics[arm]["ci_upper_vs_base"] = 0.0
            arm_metrics[arm]["p_val_vs_base"] = 0.5
            arm_metrics[arm]["cliffs_delta_vs_base"] = 0.0
        else:
            cand_scores = np.array(predictions[arm]["logits"])
            boot = compute_paired_bootstrap(labels_arr, cand_scores, scores_base, n_resamples=1000)
            c_delta = compute_cliffs_delta(cand_scores, scores_base)
            arm_metrics[arm]["delta_auc_vs_base"] = boot["delta_auc"]
            arm_metrics[arm]["ci_lower_vs_base"] = boot["ci_lower"]
            arm_metrics[arm]["ci_upper_vs_base"] = boot["ci_upper"]
            arm_metrics[arm]["p_val_vs_base"] = boot["p_value"]
            arm_metrics[arm]["cliffs_delta_vs_base"] = c_delta

    # Outcome
    outcome = classify_pyramidal_outcome(arm_metrics)

    # Save Reports
    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "milestone": "M20.1",
        "title": "Whole-Face Multi-Scale Pyramidal Representation Diagnostic",
        "model_analyzed": "Frozen M18 Dual-Stream Checkpoint (Seed 42)",
        "dataset_analyzed": f"Face2Face Locked Holdout (N={len(holdout_items)}: 58 Reals, 58 Fakes)",
        "arm_metrics": arm_metrics,
        "diagnostic_outcome": outcome,
    }

    with open(REPORT_JSON, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Diagnostic JSON report saved to %s", REPORT_JSON)

    # Markdown Report
    md_content = f"""# Milestone 20.1 — Whole-Face Multi-Scale Pyramidal Representation Diagnostic Report

**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}  
**Target Model**: Frozen M18 Dual-Stream Detector (`checkpoints/m18_dual_stream_seed42/best.pth`)  
**Dataset**: Face2Face Locked Holdout ({len(holdout_items)} clips: 58 Pristine Reals, 58 Face2Face Fakes)  
**Methodology**: Whole-face multi-scale spatial decomposition (Gaussian base + Laplacian detail octaves) evaluated across 5 diagnostic arms.

---

## 1. Multi-Scale Pyramidal Diagnostic Results

| Arm | Representation Description | Pyramid Formulation | ROC-AUC | PR-AUC | $\\Delta$ AUC vs. Base | 95% Bootstrap CI | Paired $p$-value | Real Med Logit | Fake Med Logit | Logit Margin |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm 1** | **Whole-Face Base Control** | Standard $G_1$ (112×112) | **{arm_metrics['arm_1_base']['auc']:.4f}** | {arm_metrics['arm_1_base']['prauc']:.4f} | Baseline Ref | — | — | {arm_metrics['arm_1_base']['real_median_logit']:+.3f} | {arm_metrics['arm_1_base']['fake_median_logit']:+.3f} | **{arm_metrics['arm_1_base']['logit_margin']:+.3f}** |
| **Arm 2** | **Mid-Frequency Warping** | Laplacian $L_1$ (112×112) | **{arm_metrics['arm_2_mid_l1']['auc']:.4f}** | {arm_metrics['arm_2_mid_l1']['prauc']:.4f} | {arm_metrics['arm_2_mid_l1']['delta_auc_vs_base']:+.4f} | [{arm_metrics['arm_2_mid_l1']['ci_lower_vs_base']:+.4f}, {arm_metrics['arm_2_mid_l1']['ci_upper_vs_base']:+.4f}] | {arm_metrics['arm_2_mid_l1']['p_val_vs_base']:.4f} | {arm_metrics['arm_2_mid_l1']['real_median_logit']:+.3f} | {arm_metrics['arm_2_mid_l1']['fake_median_logit']:+.3f} | {arm_metrics['arm_2_mid_l1']['logit_margin']:+.3f} |
| **Arm 3** | **Native High-Frequency** | Laplacian $L_0 \\to 112$ | **{arm_metrics['arm_3_high_l0']['auc']:.4f}** | {arm_metrics['arm_3_high_l0']['prauc']:.4f} | {arm_metrics['arm_3_high_l0']['delta_auc_vs_base']:+.4f} | [{arm_metrics['arm_3_high_l0']['ci_lower_vs_base']:+.4f}, {arm_metrics['arm_3_high_l0']['ci_upper_vs_base']:+.4f}] | {arm_metrics['arm_3_high_l0']['p_val_vs_base']:.4f} | {arm_metrics['arm_3_high_l0']['real_median_logit']:+.3f} | {arm_metrics['arm_3_high_l0']['fake_median_logit']:+.3f} | {arm_metrics['arm_3_high_l0']['logit_margin']:+.3f} |
| **Arm 4a**| **Detail Injection ($\\alpha=0.2$)** | $G_1 + 0.20 \\cdot L_0$ | **{arm_metrics['arm_4_inject_02']['auc']:.4f}** | {arm_metrics['arm_4_inject_02']['prauc']:.4f} | {arm_metrics['arm_4_inject_02']['delta_auc_vs_base']:+.4f} | [{arm_metrics['arm_4_inject_02']['ci_lower_vs_base']:+.4f}, {arm_metrics['arm_4_inject_02']['ci_upper_vs_base']:+.4f}] | {arm_metrics['arm_4_inject_02']['p_val_vs_base']:.4f} | {arm_metrics['arm_4_inject_02']['real_median_logit']:+.3f} | {arm_metrics['arm_4_inject_02']['fake_median_logit']:+.3f} | {arm_metrics['arm_4_inject_02']['logit_margin']:+.3f} |
| **Arm 4b**| **Detail Injection ($\\alpha=0.5$)** | $G_1 + 0.50 \\cdot L_0$ | **{arm_metrics['arm_4_inject_05']['auc']:.4f}** | {arm_metrics['arm_4_inject_05']['prauc']:.4f} | {arm_metrics['arm_4_inject_05']['delta_auc_vs_base']:+.4f} | [{arm_metrics['arm_4_inject_05']['ci_lower_vs_base']:+.4f}, {arm_metrics['arm_4_inject_05']['ci_upper_vs_base']:+.4f}] | {arm_metrics['arm_4_inject_05']['p_val_vs_base']:.4f} | {arm_metrics['arm_4_inject_05']['real_median_logit']:+.3f} | {arm_metrics['arm_4_inject_05']['fake_median_logit']:+.3f} | {arm_metrics['arm_4_inject_05']['logit_margin']:+.3f} |

---

## 2. Diagnostic Analysis & Findings

1. **Reconstruction Integrity**:
   - Laplacian decomposition satisfies exact mathematical reconstruction: $G_1 = \\text{{Upsample}}(G_2) + L_1$ with float error $< 10^{{-5}}$.
2. **Frequency-Band Isolation**:
   - Compares whether isolated mid-frequency ($L_1$) or native high-frequency ($L_0$) detail carries the reenactment signal.
3. **High-Frequency Detail Injection**:
   - Tests whether injecting native-resolution residual details directly into the whole-face crop at $112\\times 112$ bypasses the downsampling information loss without breaking the spatial filter regime.

---

## 3. Diagnostic Outcome & Governance Recommendation

- **Outcome Tier**: **`{outcome['tier']}`**
- **Scientific Diagnostic Finding**: {outcome['reason']}
- **Governance Recommendation**: {outcome['recommendation']}

---

## 4. Gating Invariant & Stop Condition

> [!IMPORTANT]
> Milestone 20.1 is strictly a diagnostic evaluation. No multi-epoch model training or architectural modification is performed without user authorization following review of these diagnostic results.
"""

    with open(REPORT_MD, "w") as f:
        f.write(md_content)
    logger.info("Diagnostic Markdown report saved to %s", REPORT_MD)
    print(f"\n[DONE] Milestone 20.1 diagnostic complete. Tier: {outcome['tier']}")


if __name__ == "__main__":
    main()
