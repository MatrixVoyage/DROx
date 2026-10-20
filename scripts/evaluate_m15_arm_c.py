"""Milestone 15 Arm C Multi-Gate Promotion & Evaluation Script (M15-RobustGen).

Evaluates M15 Arm C candidate against:
1. The Six Preregistered Promotion Gates (context/m15_preregistration.md):
   - Gate 1: Face2Face Transfer Gain (Delta AUC_F2F >= +0.0500, p < 0.05)
   - Gate 2: External Non-Inferiority Margin (Delta AUC_CelebDF >= -0.0150)
   - Gate 3: Clean FF++ Specificity Floor (Spec_clean >= 93.0% at tau = 0.50)
   - Gate 4: Clean Celeb-DF Specificity Floor (Spec_celeb >= 96.5% at tau = 0.50)
   - Gate 5: Compression Robustness Non-Inferiority Margin (ROC-AUC >= 0.6500 under JPEG Q=50)
   - Gate 6: Runtime Latency & VRAM (P50 <= 35.0ms, Peak VRAM <= 5.2GB)
2. Hypothesis H-M15-3 Verification:
   - Quantifies whether frequency consistency loss reduces Real -> Fake flip rate under JPEG Q=50 by > 20% relative to Arm B.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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
from src.models.m15_cross_attention_detector import (
    M15CrossAttentionDetector,
    M15DetectorOutput,
)
from src.utils.config import _dict_to_config
from src.utils.logger import setup_logger

logger = setup_logger("m15_arm_c_eval")

# Benchmark Paths
MANIFEST_MIXED = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"
MANIFEST_CELEBDF = PROJECT_ROOT / "manifests" / "celebdf_v2.csv"
MANIFEST_LOFO = PROJECT_ROOT / "manifests" / "lofo_f2f.csv"

M15_ARM_C_CKPT = PROJECT_ROOT / "checkpoints" / "m15_arm_c_seed42" / "best.pth"
M15_ARM_B_CKPT = PROJECT_ROOT / "checkpoints" / "m15_arm_b_seed42" / "best.pth"
V5_CHAMPION_CKPT = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "best.pth"
V5_CALIB_CONFIG = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "calibration_config.json"

TENSOR_DIR_M12 = PROJECT_ROOT / "processed" / "m12_full"
TENSOR_DIR_LOFO = PROJECT_ROOT / "processed" / "m13_3_lofo"
TENSOR_DIR_CELEBDF = PROJECT_ROOT / "processed" / "celebdf_v2" / "test" / "tensors"

REPORT_JSON = PROJECT_ROOT / "reports" / "m15_arm_c_evaluation_report.json"
REPORT_MD = PROJECT_ROOT / "reports" / "m15_arm_c_evaluation_report.md"


def sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_tensor(video_id: str, split: str = "test") -> Optional[Path]:
    """Find preprocessed tensor across known directories."""
    candidates = [
        TENSOR_DIR_M12 / split / "tensors" / f"{video_id}.pt",
        TENSOR_DIR_LOFO / split / "tensors" / f"{video_id}.pt",
        TENSOR_DIR_CELEBDF / f"{video_id}.pt",
        TENSOR_DIR_M12 / "val" / "tensors" / f"{video_id}.pt",
        TENSOR_DIR_M12 / "train" / "tensors" / f"{video_id}.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def apply_simulated_jpeg(tensor: torch.Tensor, quality: int = 50) -> torch.Tensor:
    """Apply simulated JPEG compression to (16, 3, 112, 112) normalized tensor."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = tensor.cpu().numpy()
    out_arr = np.zeros_like(arr)

    for t in range(arr.shape[0]):
        frame = arr[t].transpose(1, 2, 0)
        frame = (frame * std + mean) * 255.0
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, enc = cv2.imencode(".jpg", frame, encode_param)
        if success:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        else:
            dec = frame

        norm_frame = dec.astype(np.float32) / 255.0
        norm_frame = (norm_frame - mean) / std
        out_arr[t] = norm_frame.transpose(2, 0, 1)

    return torch.from_numpy(out_arr).to(tensor.device, dtype=tensor.dtype)


def fit_temperature_scaling(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Fit temperature parameter T > 0 by minimizing Negative Log-Likelihood on validation set."""
    y = val_labels.astype(np.float64)
    z = val_logits.astype(np.float64)

    def nll(T: float) -> float:
        T = max(1e-4, float(T))
        p = 1.0 / (1.0 + np.exp(-z / T))
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


def derive_spec90_threshold(clean_probs: np.ndarray, target_spec: float = 0.90) -> float:
    """Derive operating threshold tau* guaranteeing clean specificity >= target_spec."""
    sorted_probs = np.sort(clean_probs)
    rank = int(np.floor((1.0 - target_spec) * len(sorted_probs)))
    rank = min(max(0, rank), len(sorted_probs) - 1)
    return float(sorted_probs[rank])


def compute_paired_bootstrap(
    y_true: np.ndarray,
    scores_cand: np.ndarray,
    scores_base: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute paired bootstrap distribution of delta AUC = AUC(cand) - AUC(base)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []

    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_cand = roc_auc_score(y_b, scores_cand[idx])
        auc_base = roc_auc_score(y_b, scores_base[idx])
        deltas.append(auc_cand - auc_base)

    deltas = np.array(deltas)
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    mean_delta = float(np.mean(deltas))

    p_value = float(np.mean(deltas <= 0.0) * 2.0)
    p_value = min(1.0, max(0.001, p_value))

    return {
        "mean_delta": round(mean_delta, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="M15 Arm C (M15-RobustGen) Multi-Gate Promotion Evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Evaluating M15 Arm C on device: {device}")

    assert M15_ARM_C_CKPT.is_file(), f"M15 Arm C checkpoint not found: {M15_ARM_C_CKPT}"
    assert V5_CHAMPION_CKPT.is_file(), f"v5 champion checkpoint not found: {V5_CHAMPION_CKPT}"

    m15_c_hash = sha256_file(M15_ARM_C_CKPT)
    v5_hash = sha256_file(V5_CHAMPION_CKPT)
    logger.info(f"M15 Arm C SHA-256: {m15_c_hash}")
    logger.info(f"v5 Champion SHA-256: {v5_hash}")

    # 1. Load M15 Arm C Model
    ckpt_c = torch.load(M15_ARM_C_CKPT, map_location=device, weights_only=False)
    m12_cfg = _dict_to_config(ckpt_c["config"]) if isinstance(ckpt_c["config"], dict) else ckpt_c["config"]

    base_c = DeepfakeDetector(config=m12_cfg.model).to(device)
    model_c = M15CrossAttentionDetector(
        base_detector=base_c,
        num_families=5,
        aux_hidden_dim=512,
        aux_dropout=0.5,
        d_attn=256,
        nhead=4,
        gate_bottleneck=128,
        mid_in_channels=112,
        enable_local_refinement=False,
    ).to(device)
    model_c.load_state_dict(ckpt_c["model_state_dict"])
    model_c.eval()

    # 2. Load Arm B Model (for H-M15-3 relative flip rate comparison)
    has_arm_b = M15_ARM_B_CKPT.is_file()
    model_b = None
    if has_arm_b:
        ckpt_b = torch.load(M15_ARM_B_CKPT, map_location=device, weights_only=False)
        base_b = DeepfakeDetector(config=m12_cfg.model).to(device)
        model_b = M15CrossAttentionDetector(
            base_detector=base_b,
            num_families=5,
            aux_hidden_dim=512,
            aux_dropout=0.5,
            d_attn=256,
            nhead=4,
            gate_bottleneck=128,
            mid_in_channels=112,
            enable_local_refinement=False,
        ).to(device)
        model_b.load_state_dict(ckpt_b["model_state_dict"])
        model_b.eval()

    # 3. Load v5-family-invariant Champion (Frozen Reference Control)
    ckpt_v5 = torch.load(V5_CHAMPION_CKPT, map_location=device, weights_only=False)
    base_v5 = DeepfakeDetector(config=m12_cfg.model).to(device)
    model_v5 = FamilyInvariantDeepfakeDetector(
        base_detector=base_v5,
        num_families=5,
        aux_hidden_dim=512,
        aux_dropout=0.5,
        freq_dim=64,
        texture_dim=64,
        fusion_dim=512,
        enable_mid_head=True,
        mid_in_channels=112,
        mid_hidden_dim=128,
    ).to(device)
    model_v5.load_state_dict(ckpt_v5["model_state_dict"])
    model_v5.eval()

    with open(V5_CALIB_CONFIG, "r") as f:
        v5_calib = json.load(f)
    v5_T = float(v5_calib.get("temperature_scaling", {}).get("temperature", 2.0502))
    v5_tau_star = float(v5_calib.get("operating_thresholds", {}).get("high_sensitivity_spec90", 0.1260))

    # Phase 1: Derive M15-C Calibration on M15-Val
    logger.info("Phase 1: Deriving calibration and thresholds on M15-Val...")
    lofo_df = pd.read_csv(MANIFEST_LOFO)
    val_ffpp = lofo_df[(lofo_df["split"] == "val") & (lofo_df["dataset"] == "ffpp")].copy().reset_index(drop=True)

    val_records = []
    for _, row in val_ffpp.iterrows():
        vid = str(row["video_id"])
        tp = find_tensor(vid, split="val")
        if tp is not None:
            val_records.append((tp, int(row["label"])))

    val_logits_c = []
    val_labels = []

    with torch.no_grad():
        for tp, lbl in val_records:
            tensor = torch.load(tp, map_location=device, weights_only=False)
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)
            out = model_c(tensor, return_dict=True)
            val_logits_c.append(out.primary_logits.item())
            val_labels.append(lbl)

    val_logits_arr = np.array(val_logits_c)
    val_labels_arr = np.array(val_labels)

    T_c = fit_temperature_scaling(val_logits_arr, val_labels_arr)
    val_clean_mask = val_labels_arr == 0
    val_clean_cal_probs = 1.0 / (1.0 + np.exp(-val_logits_arr[val_clean_mask] / T_c))
    tau_star_c = derive_spec90_threshold(val_clean_cal_probs, target_spec=0.90)

    logger.info(f"Derived M15-C Calibration: T = {T_c:.4f} | tau* = {tau_star_c:.4f}")

    # Phase 2: Gate 1 & Gate 3 — Face2Face Holdout Benchmark & Clean FF++ Specificity
    logger.info("Phase 2: Evaluating Gate 1 (Face2Face) & Gate 3 (Clean FF++ Specificity)...")
    mixed_df = pd.read_csv(MANIFEST_MIXED)

    f2f_test = mixed_df[(mixed_df["split"] == "test") & (mixed_df["manipulation_type"] == "Face2Face")].copy().reset_index(drop=True)
    clean_test = mixed_df[(mixed_df["split"] == "test") & (mixed_df["label"] == 0)].copy().reset_index(drop=True)

    f2f_items = [(find_tensor(str(r["video_id"]), split="test"), 1) for _, r in f2f_test.iterrows()]
    f2f_items = [item for item in f2f_items if item[0] is not None]

    clean_items = [(find_tensor(str(r["video_id"]), split="test"), 0) for _, r in clean_test.iterrows()]
    clean_items = [item for item in clean_items if item[0] is not None]

    benchmark_items = clean_items + f2f_items
    bench_labels = []
    scores_c_cal = []
    scores_v5_cal = []

    with torch.no_grad():
        for tp, lbl in benchmark_items:
            tensor = torch.load(tp, map_location=device, weights_only=False)
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)

            out_c = model_c(tensor, return_dict=True)
            p_cal_c = torch.sigmoid(out_c.primary_logits / T_c).item()

            out_v5 = model_v5(tensor, return_dict=True)
            p_cal_v5 = torch.sigmoid(out_v5.primary_logits / v5_T).item()

            bench_labels.append(lbl)
            scores_c_cal.append(p_cal_c)
            scores_v5_cal.append(p_cal_v5)

    bench_labels_arr = np.array(bench_labels)
    scores_c_arr = np.array(scores_c_cal)
    scores_v5_arr = np.array(scores_v5_cal)

    auc_c_f2f = float(roc_auc_score(bench_labels_arr, scores_c_arr))
    auc_v5_f2f = float(roc_auc_score(bench_labels_arr, scores_v5_arr))
    delta_auc_f2f = auc_c_f2f - auc_v5_f2f

    boot_res = compute_paired_bootstrap(
        y_true=bench_labels_arr,
        scores_cand=scores_c_arr,
        scores_base=scores_v5_arr,
        n_resamples=1000,
        seed=42,
    )

    f2f_mask = bench_labels_arr == 1
    f2f_det_std_c = int(np.sum(scores_c_arr[f2f_mask] >= 0.50))
    f2f_det_spec90_c = int(np.sum(scores_c_arr[f2f_mask] >= tau_star_c))

    clean_mask = bench_labels_arr == 0
    clean_correct_c = int(np.sum(scores_c_arr[clean_mask] < 0.50))
    clean_spec_c = clean_correct_c / len(clean_items)
    clean_false_alarms_c = len(clean_items) - clean_correct_c

    logger.info(f"F2F AUC: M15-C={auc_c_f2f:.4f} | v5={auc_v5_f2f:.4f} | Delta={delta_auc_f2f:+.4f} (p={boot_res['p_value']:.4f})")
    logger.info(f"F2F Det Rate (tau=0.50): {f2f_det_std_c/len(f2f_items)*100:.2f}% | (tau*={tau_star_c:.4f}): {f2f_det_spec90_c/len(f2f_items)*100:.2f}%")
    logger.info(f"Clean FF++ Specificity: {clean_spec_c*100:.2f}% (False Alarms: {clean_false_alarms_c}/{len(clean_items)})")

    # Phase 3: Gate 2 & Gate 4 — Celeb-DF v2 Benchmark
    logger.info("Phase 3: Evaluating Gate 2 (Celeb-DF Generalization) & Gate 4 (Clean Celeb Specificity)...")
    celeb_df = pd.read_csv(MANIFEST_CELEBDF)
    celeb_test = celeb_df[celeb_df["split"] == "test"].copy().reset_index(drop=True)

    celeb_items = []
    for _, row in celeb_test.iterrows():
        vid = str(row["video_id"])
        tp = find_tensor(vid, split="test")
        if tp is not None:
            celeb_items.append((tp, int(row["label"])))

    celeb_labels = []
    celeb_scores_c = []
    celeb_scores_v5 = []

    with torch.no_grad():
        for tp, lbl in celeb_items:
            tensor = torch.load(tp, map_location=device, weights_only=False)
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)

            out_c = model_c(tensor, return_dict=True)
            p_cal_c = torch.sigmoid(out_c.primary_logits / T_c).item()

            out_v5 = model_v5(tensor, return_dict=True)
            p_cal_v5 = torch.sigmoid(out_v5.primary_logits / v5_T).item()

            celeb_labels.append(lbl)
            celeb_scores_c.append(p_cal_c)
            celeb_scores_v5.append(p_cal_v5)

    celeb_labels_arr = np.array(celeb_labels)
    celeb_c_arr = np.array(celeb_scores_c)
    celeb_v5_arr = np.array(celeb_scores_v5)

    auc_c_celeb = float(roc_auc_score(celeb_labels_arr, celeb_c_arr))
    auc_v5_celeb = float(roc_auc_score(celeb_labels_arr, celeb_v5_arr))
    delta_auc_celeb = auc_c_celeb - auc_v5_celeb

    celeb_clean_mask = celeb_labels_arr == 0
    n_celeb_clean = int(np.sum(celeb_clean_mask))
    celeb_correct_c = int(np.sum(celeb_c_arr[celeb_clean_mask] < 0.50))
    celeb_spec_c = celeb_correct_c / max(1, n_celeb_clean)
    celeb_false_alarms_c = n_celeb_clean - celeb_correct_c

    logger.info(f"Celeb-DF AUC: M15-C={auc_c_celeb:.4f} | v5={auc_v5_celeb:.4f} | Delta={delta_auc_celeb:+.4f}")
    logger.info(f"Celeb-DF Clean Specificity: {celeb_spec_c*100:.2f}% (False Alarms: {celeb_false_alarms_c}/{n_celeb_clean})")

    # Phase 4: Gate 5 — Compression Robustness (JPEG Q=50) & Hypothesis H-M15-3 Flip Test
    logger.info("Phase 4: Evaluating Gate 5 (Compression Robustness) & Hypothesis H-M15-3...")
    q50_labels = []
    q50_scores_c = []
    q50_scores_b = []

    pristine_scores_clean_c = []
    q50_scores_clean_c = []
    pristine_scores_clean_b = []
    q50_scores_clean_b = []

    with torch.no_grad():
        for tp, lbl in benchmark_items:
            tensor = torch.load(tp, map_location=device, weights_only=False)
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)

            # Clean inference
            out_c_clean = model_c(tensor, return_dict=True)
            p_c_clean = torch.sigmoid(out_c_clean.primary_logits / T_c).item()

            if model_b is not None:
                out_b_clean = model_b(tensor, return_dict=True)
                p_b_clean = out_b_clean.p_fused.item()

            # Simulated JPEG Q=50
            degraded = apply_simulated_jpeg(tensor.squeeze(0), quality=50).unsqueeze(0)
            out_c_q50 = model_c(degraded, return_dict=True)
            p_c_q50 = torch.sigmoid(out_c_q50.primary_logits / T_c).item()

            q50_labels.append(lbl)
            q50_scores_c.append(p_c_q50)

            if model_b is not None:
                out_b_q50 = model_b(degraded, return_dict=True)
                p_b_q50 = out_b_q50.p_fused.item()
                q50_scores_b.append(p_b_q50)

            if lbl == 0:
                pristine_scores_clean_c.append(p_c_clean)
                q50_scores_clean_c.append(p_c_q50)
                if model_b is not None:
                    pristine_scores_clean_b.append(p_b_clean)
                    q50_scores_clean_b.append(p_b_q50)

    auc_q50_c = float(roc_auc_score(q50_labels, q50_scores_c))
    logger.info(f"JPEG Q=50 ROC-AUC (Arm C): {auc_q50_c:.4f} (Ceiling threshold: >= 0.6500)")

    # Hypothesis H-M15-3: Real -> Fake Flip Rate Analysis under JPEG Q=50
    # Flip occurs when clean real was correctly classified (p < 0.50) but flips to Fake (p >= 0.50) under Q=50
    clean_c_pristine = np.array(pristine_scores_clean_c)
    clean_c_q50 = np.array(q50_scores_clean_c)
    correct_clean_mask_c = clean_c_pristine < 0.50
    flips_c = int(np.sum((clean_c_pristine < 0.50) & (clean_c_q50 >= 0.50)))
    total_eligible_c = int(np.sum(correct_clean_mask_c))
    flip_rate_c = flips_c / max(1, total_eligible_c)

    if model_b is not None and len(pristine_scores_clean_b) > 0:
        clean_b_pristine = np.array(pristine_scores_clean_b)
        clean_b_q50 = np.array(q50_scores_clean_b)
        correct_clean_mask_b = clean_b_pristine < 0.50
        flips_b = int(np.sum((clean_b_pristine < 0.50) & (clean_b_q50 >= 0.50)))
        total_eligible_b = int(np.sum(correct_clean_mask_b))
        flip_rate_b = flips_b / max(1, total_eligible_b)
        flip_reduction = (flip_rate_b - flip_rate_c) / max(1e-6, flip_rate_b)
    else:
        flip_rate_b = 0.0
        flips_b = 0
        total_eligible_b = 0
        flip_reduction = 0.0

    logger.info(f"H-M15-3 Flip Test: Arm C Flips={flips_c}/{total_eligible_c} ({flip_rate_c*100:.2f}%) vs Arm B Flips={flips_b}/{total_eligible_b} ({flip_rate_b*100:.2f}%)")
    logger.info(f"H-M15-3 Flip Rate Reduction: {flip_reduction*100:.2f}% (Target: > 20.0%)")

    # Phase 5: Gate 6 — Runtime Latency & VRAM Benchmarking
    logger.info("Phase 5: Evaluating Gate 6 (Runtime Latency P50 & Peak VRAM)...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    dummy_input = torch.randn(1, 16, 3, 112, 112, device=device)

    for _ in range(10):
        with torch.no_grad():
            _ = model_c(dummy_input, return_dict=True)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_c(dummy_input, return_dict=True)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    p50_latency = float(latencies[len(latencies) // 2])
    peak_vram_gb = float(torch.cuda.max_memory_allocated() / (1024**3))

    # Phase 6: Synthesis Across the Six Preregistered Promotion Gates
    logger.info("Phase 6: Synthesizing Promotion Gate Decisions...")

    gate1_pass = bool(delta_auc_f2f >= 0.0500 and boot_res["p_value"] < 0.05)
    gate2_pass = bool(delta_auc_celeb >= -0.0150)
    gate3_pass = bool(clean_spec_c >= 0.9300)
    gate4_pass = bool(celeb_spec_c >= 0.9650)
    gate5_pass = bool(auc_q50_c >= 0.6500)
    gate6_pass = bool(p50_latency <= 35.0 and peak_vram_gb <= 5.20)

    all_gates_pass = bool(
        gate1_pass and gate2_pass and gate3_pass and gate4_pass and gate5_pass and gate6_pass
    )
    promotion_verdict = "PROMOTE_TO_CHAMPION" if all_gates_pass else "RETAIN_M13_5_FROZEN"

    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "name": "M15-RobustGen",
            "arm": "Arm C",
            "checkpoint_path": str(M15_ARM_C_CKPT),
            "sha256": m15_c_hash,
            "calibration_temperature_T": round(T_c, 4),
            "spec90_threshold_tau_star": round(tau_star_c, 4),
        },
        "reference_control": {
            "name": "v5-family-invariant",
            "arm": "Arm A",
            "checkpoint_path": str(V5_CHAMPION_CKPT),
            "sha256": v5_hash,
            "calibration_temperature_T": round(v5_T, 4),
            "spec90_threshold_tau_star": round(v5_tau_star, 4),
        },
        "hypothesis_h_m15_3": {
            "hypothesis": "Frequency consistency loss reduces Real -> Fake flip rate under JPEG Q=50 by > 20%",
            "flips_arm_c": flips_c,
            "flips_arm_b": flips_b,
            "flip_rate_arm_c": round(flip_rate_c, 4),
            "flip_rate_arm_b": round(flip_rate_b, 4),
            "flip_reduction_ratio": round(flip_reduction, 4),
            "verdict": "SUPPORTED" if flip_reduction >= 0.20 else "UNSUPPORTED",
        },
        "gates": {
            "gate1_face2face_transfer": {
                "metric": "Delta ROC-AUC on Face2Face",
                "target": ">= +0.0500 (p < 0.05)",
                "auc_candidate": round(auc_c_f2f, 4),
                "auc_reference": round(auc_v5_f2f, 4),
                "delta_auc": round(delta_auc_f2f, 4),
                "bootstrap_p_value": boot_res["p_value"],
                "status": "PASS" if gate1_pass else "FAIL",
            },
            "gate2_external_non_inferiority": {
                "metric": "Delta ROC-AUC on Celeb-DF v2",
                "target": ">= -0.0150",
                "auc_candidate": round(auc_c_celeb, 4),
                "auc_reference": round(auc_v5_celeb, 4),
                "delta_auc": round(delta_auc_celeb, 4),
                "status": "PASS" if gate2_pass else "FAIL",
            },
            "gate3_clean_ffpp_specificity": {
                "metric": "Clean Real Specificity (tau=0.50)",
                "target": ">= 93.00%",
                "specificity": round(clean_spec_c, 4),
                "false_alarms": clean_false_alarms_c,
                "total_clean": len(clean_items),
                "status": "PASS" if gate3_pass else "FAIL",
            },
            "gate4_clean_celeb_specificity": {
                "metric": "Clean Celebrity Specificity (tau=0.50)",
                "target": ">= 96.50%",
                "specificity": round(celeb_spec_c, 4),
                "false_alarms": celeb_false_alarms_c,
                "total_clean": n_celeb_clean,
                "status": "PASS" if gate4_pass else "FAIL",
            },
            "gate5_compression_robustness": {
                "metric": "ROC-AUC under JPEG Q=50",
                "target": ">= 0.6500",
                "auc_q50": round(auc_q50_c, 4),
                "status": "PASS" if gate5_pass else "FAIL",
            },
            "gate6_runtime_and_vram": {
                "metric": "P50 Latency & Peak VRAM",
                "target": "P50 <= 35.0ms, VRAM <= 5.20GB",
                "p50_latency_ms": round(p50_latency, 2),
                "peak_vram_gb": round(peak_vram_gb, 2),
                "status": "PASS" if gate6_pass else "FAIL",
            },
        },
        "verdict": {
            "all_gates_pass": all_gates_pass,
            "decision": promotion_verdict,
        },
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    md_content = f"""# Milestone 15 Arm C Evaluation Report: Robustness & Domain Generalization

**Evaluation Date**: `{report_data['timestamp']}`  
**Governing Standard**: `context/m15_preregistration.md`  
**Arm Designation**: **Arm C (`M15-RobustGen`)**  
**Promotion Verdict**: **`{promotion_verdict}`**  

---

## 1. Executive Summary & Promotion Verdict

Milestone 15 executed the preregistered evaluation of **Arm C (`M15-RobustGen`)** (cross-attention + multi-scale frequency consistency loss + compression jitter) against the **frozen `v5-family-invariant` control (Arm A)**.

| Promotion Gate | Metric / Target | Candidate Score | Reference Score | Delta / Margin | Status |
|---|---|:---:|:---:|:---:|:---:|
| **Gate 1: Face2Face Gain** | $\\Delta \\text{{AUC}}_{{\\text{{F2F}}}} \\ge +0.0500, p < 0.05$ | `{auc_c_f2f:.4f}` | `{auc_v5_f2f:.4f}` | `{delta_auc_f2f:+.4f}` ($p={boot_res['p_value']:.4f}$) | **{report_data['gates']['gate1_face2face_transfer']['status']}** |
| **Gate 2: External Generalization** | $\\Delta \\text{{AUC}}_{{\\text{{CelebDF}}}} \\ge -0.0150$ | `{auc_c_celeb:.4f}` | `{auc_v5_celeb:.4f}` | `{delta_auc_celeb:+.4f}` | **{report_data['gates']['gate2_external_non_inferiority']['status']}** |
| **Gate 3: Clean FF++ Specificity** | $\\text{{Spec}}_{{\\text{{clean}}}} \\ge 93.00\\%$ ($\\tau=0.50$) | `{clean_spec_c*100:.2f}\\%$` (`{clean_false_alarms_c}/{len(clean_items)}` FA) | `93.88\\%` | Floor: $93.00\\%$ | **{report_data['gates']['gate3_clean_ffpp_specificity']['status']}** |
| **Gate 4: Clean Celeb Specificity** | $\\text{{Spec}}_{{\\text{{celeb}}}} \\ge 96.50\\%$ ($\\tau=0.50$) | `{celeb_spec_c*100:.2f}\\%$` (`{celeb_false_alarms_c}/{n_celeb_clean}` FA) | `97.75\\%` | Floor: $96.50\\%$ | **{report_data['gates']['gate4_clean_celeb_specificity']['status']}** |
| **Gate 5: Compression Robustness** | $\\text{{ROC-AUC}}(\\text{{JPEG }} Q=50) \\ge 0.6500$ | `{auc_q50_c:.4f}` | `0.6360` | Floor: $0.6500$ | **{report_data['gates']['gate5_compression_robustness']['status']}** |
| **Gate 6: Runtime & VRAM** | $P50 \\le 35.0\\text{{ms}}, \\text{{VRAM}} \\le 5.20\\text{{GB}}$ | `{p50_latency:.2f}\\text{{ ms}}, {peak_vram_gb:.2f}\\text{{ GB}}` | `24.95\\text{{ ms}}, 4.85\\text{{ GB}}` | Within Budgets | **{report_data['gates']['gate6_runtime_and_vram']['status']}** |

**Final Decision**: `{promotion_verdict}`  
{' All six promotion gates passed. Candidate qualifies for production promotion.' if all_gates_pass else '> **Governing Action**: Because one or more promotion gates did not meet the preregistered non-inferiority margins, `v5-family-invariant` is RETAINED as the active production champion. The candidate is preserved as an empirical research checkpoint.'}

---

## 2. Hypothesis H-M15-3 (Compression & Frequency Regularization) Findings

- **Hypothesis**: Penalizing high-frequency representation variance under simulated DCT compression during training prevents false alarms on compression blocking artifacts.
- **Arm C Real $\\to$ Fake Flips under JPEG Q=50**: `{flips_c}/{total_eligible_c}` ({flip_rate_c*100:.2f}%)
- **Arm B Real $\\to$ Fake Flips under JPEG Q=50**: `{flips_b}/{total_eligible_b}` ({flip_rate_b*100:.2f}%)
- **Relative Flip Rate Reduction**: **{flip_reduction*100:.2f}%** (Preregistered Target: $> 20.0\\%$)
- **Hypothesis Status**: **{report_data['hypothesis_h_m15_3']['verdict']}**

---

## 3. Cryptographic Provenance & Audit Trail
- **Arm C Checkpoint**: `{M15_ARM_C_CKPT}`
- **Arm C Checkpoint SHA-256**: `{m15_c_hash}`
- **v5 Champion Checkpoint**: `{V5_CHAMPION_CKPT}`
- **v5 Champion SHA-256**: `{v5_hash}`
- **Derived Temperature $T_{{\\text{{M15-C}}}}$**: `{T_c:.4f}`
- **Derived Spec90 Threshold $\\tau^*_{{\\text{{M15-C}}}}$**: `{tau_star_c:.4f}`
"""

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Evaluation report saved to: {REPORT_MD} and {REPORT_JSON}")
    logger.info(f"Final Promotion Decision: {promotion_verdict}")


if __name__ == "__main__":
    main()
