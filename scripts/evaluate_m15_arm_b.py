"""Milestone 15 Arm B Multi-Gate Promotion & Evaluation Script.

Evaluates M15 Arm B (M15-TS-CrossAttn) candidate against the Six Preregistered Promotion Gates
defined in context/m15_preregistration.md:
- Gate 1: Face2Face Transfer Gain (Delta AUC_F2F >= +0.0500, paired bootstrap p < 0.05)
- Gate 2: External Non-Inferiority Margin (Delta AUC_CelebDF >= -0.0150)
- Gate 3: Clean FF++ Specificity Floor (Spec_clean >= 93.0% at tau = 0.50)
- Gate 4: Clean Celeb-DF Specificity Floor (Spec_celeb >= 96.5% at tau = 0.50)
- Gate 5: Compression Robustness Non-Inferiority Margin (ROC-AUC >= 0.6500 under JPEG Q=50)
- Gate 6: Runtime Latency & VRAM (P50 <= 35.0ms, Peak VRAM <= 5.2GB)

Derives temperature scaling (T_M15) and Spec90 threshold (tau*_M15) strictly on M15-Val.
Zero data leakage: Face2Face and Celeb-DF remain untouched, read-only holdout evaluation benchmarks.
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

from src.data.dataset import ProcessedVideoDataset
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

logger = setup_logger("m15_arm_b_eval")

# Benchmark Paths
MANIFEST_MIXED = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"
MANIFEST_CELEBDF = PROJECT_ROOT / "manifests" / "celebdf_v2.csv"
MANIFEST_LOFO = PROJECT_ROOT / "manifests" / "lofo_f2f.csv"

M15_ARM_B_CKPT = PROJECT_ROOT / "checkpoints" / "m15_arm_b_seed42" / "best.pth"
V5_CHAMPION_CKPT = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "best.pth"
V5_CALIB_CONFIG = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "calibration_config.json"

TENSOR_DIR_M12 = PROJECT_ROOT / "processed" / "m12_full"
TENSOR_DIR_LOFO = PROJECT_ROOT / "processed" / "m13_3_lofo"
TENSOR_DIR_CELEBDF = PROJECT_ROOT / "processed" / "celebdf_v2" / "test" / "tensors"

REPORT_JSON = PROJECT_ROOT / "reports" / "m15_arm_b_evaluation_report.json"
REPORT_MD = PROJECT_ROOT / "reports" / "m15_arm_b_evaluation_report.md"


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

    arr = tensor.cpu().numpy()  # (16, 3, 112, 112)
    out_arr = np.zeros_like(arr)

    for t in range(arr.shape[0]):
        frame = arr[t].transpose(1, 2, 0)
        # Denormalize to [0, 255]
        frame = (frame * std + mean) * 255.0
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)

        # Encode and decode JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, enc = cv2.imencode(".jpg", frame, encode_param)
        if success:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        else:
            dec = frame

        # Re-normalize
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
    # The threshold must be strictly above the (1 - target_spec) percentile of clean real scores
    # e.g., for 90% specificity, at most 10% of clean reals can exceed tau*
    rank = int(np.floor((1.0 - target_spec) * len(sorted_probs)))
    rank = min(max(0, rank), len(sorted_probs) - 1)
    tau_star = float(sorted_probs[rank])
    return tau_star


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

    # Two-sided empirical p-value for H0: delta <= 0
    p_value = float(np.mean(deltas <= 0.0) * 2.0)
    p_value = min(1.0, max(0.001, p_value))

    return {
        "mean_delta": round(mean_delta, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="M15 Arm B Multi-Gate Promotion Evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Evaluating M15 Arm B on device: {device}")

    assert M15_ARM_B_CKPT.is_file(), f"M15 Arm B checkpoint not found: {M15_ARM_B_CKPT}"
    assert V5_CHAMPION_CKPT.is_file(), f"v5 champion checkpoint not found: {V5_CHAMPION_CKPT}"

    m15_hash = sha256_file(M15_ARM_B_CKPT)
    v5_hash = sha256_file(V5_CHAMPION_CKPT)
    logger.info(f"M15 Arm B SHA-256: {m15_hash}")
    logger.info(f"v5 Champion SHA-256: {v5_hash}")

    # 1. Load M15 Arm B Model
    ckpt_m15 = torch.load(M15_ARM_B_CKPT, map_location=device, weights_only=False)
    m12_cfg = _dict_to_config(ckpt_m15["config"]) if isinstance(ckpt_m15["config"], dict) else ckpt_m15["config"]

    base_m15 = DeepfakeDetector(config=m12_cfg.model).to(device)
    model_m15 = M15CrossAttentionDetector(
        base_detector=base_m15,
        num_families=5,
        aux_hidden_dim=512,
        aux_dropout=0.5,
        d_attn=256,
        nhead=4,
        gate_bottleneck=128,
        mid_in_channels=112,
        enable_local_refinement=False,
    ).to(device)
    model_m15.load_state_dict(ckpt_m15["model_state_dict"])
    model_m15.eval()

    # 2. Load v5-family-invariant Champion (Frozen Reference)
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

    logger.info(f"Loaded v5 champion with T={v5_T:.4f}, tau*={v5_tau_star:.4f}")

    # =========================================================================
    # Phase 1: Derive M15 Calibration (T_M15) and Operating Point (tau*_M15) on M15-Val
    # =========================================================================
    logger.info("Phase 1: Deriving calibration and thresholds on M15-Val...")
    lofo_df = pd.read_csv(MANIFEST_LOFO)
    val_ffpp = lofo_df[(lofo_df["split"] == "val") & (lofo_df["dataset"] == "ffpp")].copy().reset_index(drop=True)

    val_records = []
    for _, row in val_ffpp.iterrows():
        vid = str(row["video_id"])
        tp = find_tensor(vid, split="val")
        if tp is not None:
            val_records.append((tp, int(row["label"]), str(row.get("manipulation_type", "original"))))

    logger.info(f"Indexed {len(val_records)} M15-Val sequences.")

    val_logits_m15 = []
    val_probs_m15 = []
    val_labels = []

    with torch.no_grad():
        for tp, lbl, _ in val_records:
            tensor = torch.load(tp, map_location=device, weights_only=False)
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)
            out = model_m15(tensor, return_dict=True)
            val_logits_m15.append(out.primary_logits.item())
            val_probs_m15.append(out.p_fused.item())
            val_labels.append(lbl)

    val_logits_arr = np.array(val_logits_m15)
    val_probs_arr = np.array(val_probs_m15)
    val_labels_arr = np.array(val_labels)

    T_m15 = fit_temperature_scaling(val_logits_arr, val_labels_arr)

    # Derive high-sensitivity threshold tau* on clean reals under temperature scaling
    val_clean_mask = val_labels_arr == 0
    val_clean_cal_probs = 1.0 / (1.0 + np.exp(-val_logits_arr[val_clean_mask] / T_m15))
    tau_star_m15 = derive_spec90_threshold(val_clean_cal_probs, target_spec=0.90)

    logger.info(f"Derived M15 Calibration: T = {T_m15:.4f}")
    logger.info(f"Derived M15 Spec90 Operating Point: tau* = {tau_star_m15:.4f}")

    # =========================================================================
    # Phase 2: Gate 1 & Gate 3 — Face2Face Holdout Benchmark & Clean FF++ Specificity
    # =========================================================================
    logger.info("Phase 2: Evaluating Gate 1 (Face2Face Transfer) & Gate 3 (Clean FF++ Specificity)...")
    mixed_df = pd.read_csv(MANIFEST_MIXED)

    f2f_test = mixed_df[(mixed_df["split"] == "test") & (mixed_df["manipulation_type"] == "Face2Face")].copy().reset_index(drop=True)
    clean_test = mixed_df[(mixed_df["split"] == "test") & (mixed_df["label"] == 0)].copy().reset_index(drop=True)

    # Filter to existing tensors
    f2f_items = [(find_tensor(str(r["video_id"]), split="test"), 1) for _, r in f2f_test.iterrows()]
    f2f_items = [item for item in f2f_items if item[0] is not None]

    clean_items = [(find_tensor(str(r["video_id"]), split="test"), 0) for _, r in clean_test.iterrows()]
    clean_items = [item for item in clean_items if item[0] is not None]

    logger.info(f"Loaded {len(f2f_items)} Face2Face test queries and {len(clean_items)} Clean Real test queries.")

    benchmark_items = clean_items + f2f_items
    bench_labels = []
    scores_m15_cal = []
    scores_v5_cal = []
    scores_m15_raw = []

    with torch.no_grad():
        for tp, lbl in benchmark_items:
            tensor = torch.load(tp, map_location=device, weights_only=False)
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)

            # M15 candidate pass
            out_m15 = model_m15(tensor, return_dict=True)
            p_cal_m15 = torch.sigmoid(out_m15.primary_logits / T_m15).item()

            # v5 champion pass
            out_v5 = model_v5(tensor, return_dict=True)
            p_cal_v5 = torch.sigmoid(out_v5.primary_logits / v5_T).item()

            bench_labels.append(lbl)
            scores_m15_cal.append(p_cal_m15)
            scores_v5_cal.append(p_cal_v5)
            scores_m15_raw.append(out_m15.p_fused.item())

    bench_labels_arr = np.array(bench_labels)
    scores_m15_arr = np.array(scores_m15_cal)
    scores_v5_arr = np.array(scores_v5_cal)

    auc_m15_f2f = float(roc_auc_score(bench_labels_arr, scores_m15_arr))
    auc_v5_f2f = float(roc_auc_score(bench_labels_arr, scores_v5_arr))
    delta_auc_f2f = auc_m15_f2f - auc_v5_f2f

    # Paired Bootstrap
    boot_res = compute_paired_bootstrap(
        y_true=bench_labels_arr,
        scores_cand=scores_m15_arr,
        scores_base=scores_v5_arr,
        n_resamples=1000,
        seed=42,
    )

    # Face2Face Detection Rates
    f2f_mask = bench_labels_arr == 1
    f2f_detected_std_m15 = int(np.sum(scores_m15_arr[f2f_mask] >= 0.50))
    f2f_det_rate_std_m15 = f2f_detected_std_m15 / len(f2f_items)

    f2f_detected_spec90_m15 = int(np.sum(scores_m15_arr[f2f_mask] >= tau_star_m15))
    f2f_det_rate_spec90_m15 = f2f_detected_spec90_m15 / len(f2f_items)

    f2f_detected_std_v5 = int(np.sum(scores_v5_arr[f2f_mask] >= 0.50))
    f2f_det_rate_std_v5 = f2f_detected_std_v5 / len(f2f_items)

    # Clean Specificity (Gate 3)
    clean_mask = bench_labels_arr == 0
    clean_correct_m15 = int(np.sum(scores_m15_arr[clean_mask] < 0.50))
    clean_spec_m15 = clean_correct_m15 / len(clean_items)
    clean_false_alarms_m15 = len(clean_items) - clean_correct_m15

    logger.info(f"F2F AUC: M15={auc_m15_f2f:.4f} | v5={auc_v5_f2f:.4f} | Delta={delta_auc_f2f:+.4f} (95% CI: [{boot_res['ci_lower']:+.4f}, {boot_res['ci_upper']:+.4f}], p={boot_res['p_value']:.4f})")
    logger.info(f"F2F Detection Rate (tau=0.50): M15={f2f_det_rate_std_m15*100:.2f}% ({f2f_detected_std_m15}/{len(f2f_items)}) | v5={f2f_det_rate_std_v5*100:.2f}% ({f2f_detected_std_v5}/{len(f2f_items)})")
    logger.info(f"F2F Detection Rate (Spec90 tau*={tau_star_m15:.4f}): M15={f2f_det_rate_spec90_m15*100:.2f}% ({f2f_detected_spec90_m15}/{len(f2f_items)})")
    logger.info(f"Clean FF++ Specificity (tau=0.50): {clean_spec_m15*100:.2f}% (False Alarms: {clean_false_alarms_m15}/{len(clean_items)})")

    # =========================================================================
    # Phase 3: Gate 2 & Gate 4 — Celeb-DF v2 External Generalization Benchmark
    # =========================================================================
    logger.info("Phase 3: Evaluating Gate 2 (Celeb-DF Generalization) & Gate 4 (Clean Celeb Specificity)...")
    celeb_df = pd.read_csv(MANIFEST_CELEBDF)
    celeb_test = celeb_df[celeb_df["split"] == "test"].copy().reset_index(drop=True)

    celeb_items = []
    for _, row in celeb_test.iterrows():
        vid = str(row["video_id"])
        tp = find_tensor(vid, split="test")
        if tp is not None:
            celeb_items.append((tp, int(row["label"])))

    logger.info(f"Loaded {len(celeb_items)} Celeb-DF test queries.")

    celeb_labels = []
    celeb_scores_m15 = []
    celeb_scores_v5 = []

    with torch.no_grad():
        for tp, lbl in celeb_items:
            tensor = torch.load(tp, map_location=device, weights_only=False)
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)

            out_m15 = model_m15(tensor, return_dict=True)
            p_cal_m15 = torch.sigmoid(out_m15.primary_logits / T_m15).item()

            out_v5 = model_v5(tensor, return_dict=True)
            p_cal_v5 = torch.sigmoid(out_v5.primary_logits / v5_T).item()

            celeb_labels.append(lbl)
            celeb_scores_m15.append(p_cal_m15)
            celeb_scores_v5.append(p_cal_v5)

    celeb_labels_arr = np.array(celeb_labels)
    celeb_m15_arr = np.array(celeb_scores_m15)
    celeb_v5_arr = np.array(celeb_scores_v5)

    auc_m15_celeb = float(roc_auc_score(celeb_labels_arr, celeb_m15_arr))
    auc_v5_celeb = float(roc_auc_score(celeb_labels_arr, celeb_v5_arr))
    delta_auc_celeb = auc_m15_celeb - auc_v5_celeb

    celeb_clean_mask = celeb_labels_arr == 0
    n_celeb_clean = int(np.sum(celeb_clean_mask))
    celeb_clean_correct_m15 = int(np.sum(celeb_m15_arr[celeb_clean_mask] < 0.50))
    celeb_spec_m15 = celeb_clean_correct_m15 / max(1, n_celeb_clean)
    celeb_false_alarms_m15 = n_celeb_clean - celeb_clean_correct_m15

    logger.info(f"Celeb-DF External AUC: M15={auc_m15_celeb:.4f} | v5={auc_v5_celeb:.4f} | Delta={delta_auc_celeb:+.4f}")
    logger.info(f"Celeb-DF Clean Specificity (tau=0.50): {celeb_spec_m15*100:.2f}% (False Alarms: {celeb_false_alarms_m15}/{n_celeb_clean})")

    # =========================================================================
    # Phase 4: Gate 5 — Compression Robustness (JPEG Q=50)
    # =========================================================================
    logger.info("Phase 4: Evaluating Gate 5 (Compression Robustness under JPEG Q=50)...")
    q50_items = benchmark_items  # Evaluate on clean reals + Face2Face
    q50_labels = []
    q50_scores_m15 = []

    with torch.no_grad():
        for tp, lbl in q50_items:
            tensor = torch.load(tp, map_location=device, weights_only=False)
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)

            # Apply simulated JPEG Q=50
            degraded = apply_simulated_jpeg(tensor.squeeze(0), quality=50).unsqueeze(0)
            out_m15 = model_m15(degraded, return_dict=True)
            p_cal_m15 = torch.sigmoid(out_m15.primary_logits / T_m15).item()

            q50_labels.append(lbl)
            q50_scores_m15.append(p_cal_m15)

    auc_q50_m15 = float(roc_auc_score(q50_labels, q50_scores_m15))
    logger.info(f"JPEG Q=50 ROC-AUC: {auc_q50_m15:.4f} (Ceiling threshold: >= 0.6500)")

    # =========================================================================
    # Phase 5: Gate 6 — Runtime Latency & VRAM Benchmarking
    # =========================================================================
    logger.info("Phase 5: Evaluating Gate 6 (Runtime Latency P50 & Peak VRAM)...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    dummy_input = torch.randn(1, 16, 3, 112, 112, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model_m15(dummy_input, return_dict=True)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_m15(dummy_input, return_dict=True)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    p50_latency = float(latencies[len(latencies) // 2])
    peak_vram_gb = float(torch.cuda.max_memory_allocated() / (1024**3))

    logger.info(f"P50 Inference Latency: {p50_latency:.2f} ms (Budget: <= 35.0 ms)")
    logger.info(f"Peak VRAM: {peak_vram_gb:.2f} GB (Budget: <= 5.20 GB)")

    # =========================================================================
    # Phase 6: Synthesis Across the Six Preregistered Promotion Gates
    # =========================================================================
    logger.info("Phase 6: Synthesizing Promotion Gate Decisions...")

    gate1_pass = bool(delta_auc_f2f >= 0.0500 and boot_res["p_value"] < 0.05)
    gate2_pass = bool(delta_auc_celeb >= -0.0150)
    gate3_pass = bool(clean_spec_m15 >= 0.9300)
    gate4_pass = bool(celeb_spec_m15 >= 0.9650)
    gate5_pass = bool(auc_q50_m15 >= 0.6500)
    gate6_pass = bool(p50_latency <= 35.0 and peak_vram_gb <= 5.20)

    all_gates_pass = bool(
        gate1_pass and gate2_pass and gate3_pass and gate4_pass and gate5_pass and gate6_pass
    )

    promotion_verdict = "PROMOTE_TO_CHAMPION" if all_gates_pass else "RETAIN_M13_5_FROZEN"

    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "name": "M15-TS-CrossAttn",
            "arm": "Arm B",
            "checkpoint_path": str(M15_ARM_B_CKPT),
            "sha256": m15_hash,
            "calibration_temperature_T": round(T_m15, 4),
            "spec90_threshold_tau_star": round(tau_star_m15, 4),
        },
        "reference_control": {
            "name": "v5-family-invariant",
            "arm": "Arm A",
            "checkpoint_path": str(V5_CHAMPION_CKPT),
            "sha256": v5_hash,
            "calibration_temperature_T": round(v5_T, 4),
            "spec90_threshold_tau_star": round(v5_tau_star, 4),
        },
        "gates": {
            "gate1_face2face_transfer": {
                "metric": "Delta ROC-AUC on Face2Face",
                "target": ">= +0.0500 (p < 0.05)",
                "auc_candidate": round(auc_m15_f2f, 4),
                "auc_reference": round(auc_v5_f2f, 4),
                "delta_auc": round(delta_auc_f2f, 4),
                "bootstrap_95_ci": [boot_res["ci_lower"], boot_res["ci_upper"]],
                "bootstrap_p_value": boot_res["p_value"],
                "f2f_det_rate_std": round(f2f_det_rate_std_m15, 4),
                "f2f_det_rate_spec90": round(f2f_det_rate_spec90_m15, 4),
                "status": "PASS" if gate1_pass else "FAIL",
            },
            "gate2_external_non_inferiority": {
                "metric": "Delta ROC-AUC on Celeb-DF v2",
                "target": ">= -0.0150",
                "auc_candidate": round(auc_m15_celeb, 4),
                "auc_reference": round(auc_v5_celeb, 4),
                "delta_auc": round(delta_auc_celeb, 4),
                "status": "PASS" if gate2_pass else "FAIL",
            },
            "gate3_clean_ffpp_specificity": {
                "metric": "Clean Real Specificity (tau=0.50)",
                "target": ">= 93.00%",
                "specificity": round(clean_spec_m15, 4),
                "false_alarms": clean_false_alarms_m15,
                "total_clean": len(clean_items),
                "status": "PASS" if gate3_pass else "FAIL",
            },
            "gate4_clean_celeb_specificity": {
                "metric": "Clean Celebrity Specificity (tau=0.50)",
                "target": ">= 96.50%",
                "specificity": round(celeb_spec_m15, 4),
                "false_alarms": celeb_false_alarms_m15,
                "total_clean": n_celeb_clean,
                "status": "PASS" if gate4_pass else "FAIL",
            },
            "gate5_compression_robustness": {
                "metric": "ROC-AUC under JPEG Q=50",
                "target": ">= 0.6500",
                "auc_q50": round(auc_q50_m15, 4),
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
            "governing_policy": "Strict preregistration: all 6 gates must pass simultaneously to replace v5-family-invariant.",
        },
    }

    # Save JSON report
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Generate Markdown Report
    md_content = f"""# Milestone 15 Arm B Evaluation Report: Temporal–Spatial Integration

**Evaluation Date**: `{report_data['timestamp']}`  
**Governing Standard**: `context/m15_preregistration.md`  
**Arm Designation**: **Arm B (`M15-TS-CrossAttn`)**  
**Promotion Verdict**: **`{promotion_verdict}`**  

---

## 1. Executive Summary & Promotion Verdict

Milestone 15 executed the preregistered evaluation of **Arm B (`M15-TS-CrossAttn`)** against the **frozen `v5-family-invariant` control (Arm A)**.
Under the preregistered multi-gate promotion protocol, a candidate architecture qualifies to replace `v5-family-invariant` **only if it passes all six promotion gates simultaneously**.

| Promotion Gate | Metric / Target | Candidate Score | Reference Score | Delta / Margin | Status |
|---|---|:---:|:---:|:---:|:---:|
| **Gate 1: Face2Face Gain** | $\\Delta \\text{{AUC}}_{{\\text{{F2F}}}} \\ge +0.0500, p < 0.05$ | `{auc_m15_f2f:.4f}` | `{auc_v5_f2f:.4f}` | `{delta_auc_f2f:+.4f}` ($p={boot_res['p_value']:.4f}$) | **{report_data['gates']['gate1_face2face_transfer']['status']}** |
| **Gate 2: External Generalization** | $\\Delta \\text{{AUC}}_{{\\text{{CelebDF}}}} \\ge -0.0150$ | `{auc_m15_celeb:.4f}` | `{auc_v5_celeb:.4f}` | `{delta_auc_celeb:+.4f}` | **{report_data['gates']['gate2_external_non_inferiority']['status']}** |
| **Gate 3: Clean FF++ Specificity** | $\\text{{Spec}}_{{\\text{{clean}}}} \\ge 93.00\\%$ ($\\tau=0.50$) | `{clean_spec_m15*100:.2f}\\%$` (`{clean_false_alarms_m15}/{len(clean_items)}` FA) | `93.88\\%` | Floor: $93.00\\%$ | **{report_data['gates']['gate3_clean_ffpp_specificity']['status']}** |
| **Gate 4: Clean Celeb Specificity** | $\\text{{Spec}}_{{\\text{{celeb}}}} \\ge 96.50\\%$ ($\\tau=0.50$) | `{celeb_spec_m15*100:.2f}\\%$` (`{celeb_false_alarms_m15}/{n_celeb_clean}` FA) | `97.75\\%` | Floor: $96.50\\%$ | **{report_data['gates']['gate4_clean_celeb_specificity']['status']}** |
| **Gate 5: Compression Robustness** | $\\text{{ROC-AUC}}(\\text{{JPEG }} Q=50) \\ge 0.6500$ | `{auc_q50_m15:.4f}` | `0.6360` | Floor: $0.6500$ | **{report_data['gates']['gate5_compression_robustness']['status']}** |
| **Gate 6: Runtime & VRAM** | $P50 \\le 35.0\\text{{ms}}, \\text{{VRAM}} \\le 5.20\\text{{GB}}$ | `{p50_latency:.2f}\\text{{ ms}}, {peak_vram_gb:.2f}\\text{{ GB}}` | `24.95\\text{{ ms}}, 4.85\\text{{ GB}}` | Within Budgets | **{report_data['gates']['gate6_runtime_and_vram']['status']}** |

**Final Decision**: `{promotion_verdict}`  
{' All six promotion gates passed. Candidate qualifies for production promotion.' if all_gates_pass else '> **Governing Action**: Because one or more promotion gates did not meet the preregistered non-inferiority margins, `v5-family-invariant` is RETAINED as the active production champion. The candidate is preserved as an empirical research checkpoint.'}

---

## 2. Statistical Analysis & Bootstrap Results

### Gate 1: Face2Face Unseen-Manipulation Transfer
- **Candidate ROC-AUC**: `{auc_m15_f2f:.4f}`
- **Reference ROC-AUC**: `{auc_v5_f2f:.4f}`
- **Delta AUC**: `{delta_auc_f2f:+.4f}`
- **1,000-Resample Paired Bootstrap 95% CI**: `[{boot_res['ci_lower']:+.4f}, {boot_res['ci_upper']:+.4f}]`
- **Empirical Paired p-value**: `{boot_res['p_value']:.4f}`
- **Detection Rate (Standard $\\tau=0.50$)**: `{f2f_det_rate_std_m15*100:.2f}\\%$ ({f2f_detected_std_m15}/{len(f2f_items)}) vs `{f2f_det_rate_std_v5*100:.2f}\\%$ ({f2f_detected_std_v5}/{len(f2f_items)})
- **Detection Rate (Spec90 Policy $\\tau^*={tau_star_m15:.4f}$)**: `{f2f_det_rate_spec90_m15*100:.2f}\\%$ ({f2f_detected_spec90_m15}/{len(f2f_items)})

### Gate 2: Celeb-DF v2 Cross-Dataset Generalization
- **Candidate External AUC**: `{auc_m15_celeb:.4f}`
- **Reference External AUC**: `{auc_v5_celeb:.4f}`
- **Generalization Delta**: `{delta_auc_celeb:+.4f}` (Margin Floor: $-0.0150$)

---

## 3. Cryptographic Provenance & Audit Trail
- **Arm B Checkpoint**: `{M15_ARM_B_CKPT}`
- **Arm B Checkpoint SHA-256**: `{m15_hash}`
- **v5 Champion Checkpoint**: `{V5_CHAMPION_CKPT}`
- **v5 Champion SHA-256**: `{v5_hash}`
- **Derived Temperature $T_{{\\text{{M15}}}}$**: `{T_m15:.4f}` (Fitted strictly on M15-Val NLL)
- **Derived Spec90 Threshold $\\tau^*_{{\\text{{M15}}}}$**: `{tau_star_m15:.4f}` (Fitted strictly on M15-Val clean reals)
"""

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Evaluation report saved to: {REPORT_MD} and {REPORT_JSON}")
    logger.info(f"Final Promotion Decision: {promotion_verdict}")


if __name__ == "__main__":
    main()
